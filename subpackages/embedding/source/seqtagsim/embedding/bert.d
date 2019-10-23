/**
 * Copyright: Fynn Schröder, 2019
 * Author: Fynn Schröder
 * License: MIT
 */

module seqtagsim.embedding.bert;

import seqtagsim.embedding.common;
import seqtagsim.util;

import mir.ndslice;
import mir.glas.l1;
import asdf;
import zmqd;
import deimos.zmq.zmq;
import std.uuid;
import std.experimental.allocator;
import std.experimental.allocator.building_blocks.region;
import std.experimental.allocator.mallocator;
import std.typecons;
import std.parallelism;
import core.time;
import std.conv;
import std.format;
import std.algorithm : fold, max, stdMap = map, maxElement;
import std.stdio;

@extends!EmbeddingBase struct BertEmbedding
{
    mixin base;

    @disable this(this);
    enum embeddingDim = 768;

    this(bool showTokensToClient)
    {
        this.showTokensToClient = showTokensToClient;
    }

    void initialize(string serverAddress = null)
    {
        serverAddress = serverAddress == null ? "localhost" : serverAddress;
        receiveAllocator = Region!Mallocator(128 * 1024 * 1024);
        sendAllocator = Region!Mallocator(16 * 1024 * 1024);
        scope (exit)
            receiveAllocator.deallocateAll();

        uuid = randomUUID().toString();
        context = Context();
        char[256] addressBuffer;
        sender = Socket(context, SocketType.push);
        sender.linger = Duration.zero;
        sender.connect(sformat!"tcp://%s:5555"(addressBuffer, serverAddress));

        receiver = Socket(context, SocketType.sub);
        receiver.linger = Duration.zero;
        receiver.subscribe(uuid);
        receiver.connect(sformat!"tcp://%s:5556"(addressBuffer, serverAddress));

        sender.send(uuid, true);
        sender.send("SHOW_CONFIG", true);
        char[20] requestIdBuffer;
        sender.send(sformat!"%d"(requestIdBuffer, requestId++), true);
        sender.send("0");

        ubyte[36] uuidBuffer;
        immutable uuidBufferSize = receiver.receive(uuidBuffer);
        assert(uuidBuffer.length == uuidBufferSize);
        Frame config = Frame();
        receiver.receive(config);
        Asdf json = parseJson(config.data.asString, receiveAllocator);
        maxSeqLen = json["max_seq_len"].to!int;
        showTokensToClient &= json["show_tokens_to_client"].to!bool;
        immutable int poolingStrategy = json["pooling_strategy"].to!int;
        if (poolingStrategy != 0)
            assert(0, "Server must be startet with '-pooling_strategy NONE' to be able to obtain word embeddings!");
        receiver.receive(cast(ubyte[]) requestIdBuffer[]);
    }

    void beginEmbedding(size_t numberOfBatches, void function(size_t, size_t) progressCallback)
    {
        openRequests = makeArray!RequestData(Mallocator.instance, numberOfBatches);
        idOffset = requestId;
        receiverTask = scopedTask((size_t a, void function(size_t, size_t) b) {
            receiveAll(a, b);
        }, numberOfBatches, progressCallback);
        taskPool.put(receiverTask);
    }

    void normEmbeddings(string[][] sentences, Slice!(float*, 2, Contiguous) storage)
    {
        assert(storage.length!1 == embeddingDim);
        size_t[] sentenceLengths = makeArray!size_t(Mallocator.instance, sentences.stdMap!(s => s.length));
        openRequests[requestId - idOffset] = tuple(sentenceLengths, storage);
        send(sentences);
    }

    void endEmbedding()
    {
        receiverTask.yieldForce();
        dispose(Mallocator.instance, openRequests);
    }

private:

    alias RequestData = Tuple!(size_t[], Slice!(float*, 2, Contiguous));

    void send(string[][] sentences)
    {
        scope (exit)
            sendAllocator.deallocateAll;
        char[20] formatBuffer;
        auto sentenceBuffer = OutputBuffer!(ubyte, typeof(sendAllocator))(sendAllocator);
        serializeToJsonPretty!""(sentences, sentenceBuffer);
        sender.send(uuid, true);
        sender.send(sentenceBuffer.data, true);
        sender.send(sformat!"%d"(formatBuffer, requestId++), true);
        sender.send(sformat!"%d"(formatBuffer, sentences.length));
    }

    void receiveAll(size_t numberOfBatches, void function(size_t, size_t) progressCallback)
    {
        foreach(i; 0 .. numberOfBatches)
        {
            progressCallback(i, numberOfBatches);
            receive();
        }
        progressCallback(numberOfBatches, numberOfBatches);
    }

    void receive()
    {
        scope (exit)
            receiveAllocator.deallocateAll;
        ubyte[36] uuidBuf;
        immutable uuidBufferSize = receiver.receive(uuidBuf);
        assert(uuidBuf.length == uuidBufferSize);
        Frame info = Frame();
        receiver.receive(info);
        Asdf jsonInfo = parseJson(info.data.asString, receiveAllocator);
        assert(jsonInfo["dtype"] == "float32", "Only float32 is supported as embedding data type!");
        if (showTokensToClient)
            writeln("tokens: ", jsonInfo["tokens"]);
        size_t[3] shape = to!(size_t[3])(jsonInfo["shape"]);
        Slice!(float*, 3, Contiguous) slice = makeUninitSlice!float(receiveAllocator, shape);
        immutable embBufSize = receiver.receive(cast(ubyte[]) slice.field);
        assert(embBufSize == slice.elementCount * float.sizeof);
        ubyte[20] requestIdBuffer;
        immutable requestIdLength = receiver.receive(requestIdBuffer);
        size_t receivedId = requestIdBuffer[0 .. requestIdLength].asString.to!size_t;
        size_t[] sentenceLengths = openRequests[receivedId - idOffset][0];
        Slice!(float*, 2, Contiguous) storage = openRequests[receivedId - idOffset][1];
        immutable seqLen = maxSeqLen ? maxSeqLen : sentenceLengths.maxElement + 2;
        assert(shape == [sentenceLengths.length, seqLen, embeddingDim]);

        size_t i;
        foreach (j; 0 .. sentenceLengths.length)
            foreach (k; 1 .. sentenceLengths[j] + 1)
                storage[i++][] = slice[j, k] * (1f / nrm2(slice[j, k]));
        assert(i == storage.length!0);
        dispose(Mallocator.instance, sentenceLengths);
    }

    __gshared Context context;
    __gshared Socket sender;
    __gshared Socket receiver;
    string uuid;
    size_t requestId;
    __gshared size_t idOffset;
    __gshared RequestData[] openRequests;
    Task!(run, void delegate(size_t, void function(size_t, size_t)), size_t, void function(size_t, size_t)) receiverTask;
    __gshared int maxSeqLen;
    __gshared bool showTokensToClient;
    __gshared Region!Mallocator receiveAllocator;
    __gshared Region!Mallocator sendAllocator;
}

unittest
{
    BertEmbedding bert = BertEmbedding();
    bert.initialize();
    string[][] sentences = [
        ["I", "'m", "not", "sure", "how", "I", "would", "have", "handled", "it", "."],
        ["I", "had", "a", "problem", "with", "the", "tile", "in", "my", "bathroom", "coming", "apart", "."]
    ];
    size_t tokens = sentences.stdMap!(s => s.length).fold!( (a,b) => a + b)(0UL);
    auto wordEmbeddings = slice!float(tokens, BertEmbedding.embeddingDim);
    stderr.write("Fetching word embeddings for ", tokens, " tokens...");
    bert.beginEmbedding(1, (a,b) => stderr.writeln(a," ",b));
    bert.normEmbeddings(sentences, wordEmbeddings);
    bert.endEmbedding();
    stderr.writeln("Done!");
    writeln("Cosine distance ", 1f - cosineSimilarity(wordEmbeddings[3], wordEmbeddings[9]));
}
