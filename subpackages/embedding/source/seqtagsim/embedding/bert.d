/**
 * Copyright: Fynn Schröder, 2019
 * Author: Fynn Schröder
 * License: MIT
 */

module seqtagsim.embedding.bert;

import seqtagsim.embedding.common;
import seqtagsim.util;

import asdf;
import deimos.zmq.zmq;
import mir.ndslice;
import mir.glas.l1;
import zmqd;

import core.time;
import std.uuid;
import std.experimental.allocator;
import std.experimental.allocator.building_blocks.region;
import std.experimental.allocator.mallocator;
import std.typecons;
import std.parallelism;
import std.conv;
import std.format;
import std.algorithm : fold, max, stdMap = map, maxElement;
import std.stdio;

/// D client to access bert-as-service (https://github.com/hanxiao/bert-as-service)
@extends!EmbeddingBase struct BertEmbedding
{
    mixin base;

    @disable this(this);
    __gshared size_t embeddingDim;

    /// Creates a new instance with the given settings
    this(bool showTokensToClient, Duration timeout)
    {
        this.showTokensToClient = showTokensToClient;
        this.timeout = timeout;
    }

    /// Initializes the connection to the server
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
        sender.sendTimeout = timeout;
        sender.connect(sformat!"tcp://%s:5555"(addressBuffer, serverAddress));

        receiver = Socket(context, SocketType.sub);
        receiver.linger = Duration.zero;
        receiver.receiveTimeout = timeout;
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
        obtainEmbeddingDimension();
    }

    /// Starts the embedding processing for the provided number of batches
    void beginEmbedding(size_t numberOfBatches, bool normalize = true, void delegate(size_t, size_t) progressCallback = null)
    {
        openRequests = makeArray!RequestData(Mallocator.instance, numberOfBatches);
        idOffset = requestId;
        receiverTask = scopedTask((size_t a, bool n, void delegate(size_t, size_t) b) { receiveAll(a, n, b); },
                numberOfBatches, normalize, progressCallback);
        taskPool.put(receiverTask);
    }

    /// Embeds a single batch asynchronously
    void embed(string[][] sentences, Slice!(float*, 2) storage)
    {
        assert(storage.length!1 == embeddingDim);
        size_t[] sentenceLengths = makeArray!size_t(Mallocator.instance, sentences.stdMap!(s => s.length));
        openRequests[requestId - idOffset] = tuple(sentenceLengths, storage);
        send(sentences);
    }

    /// Waits for the asynchronous embedding processing to complete
    void endEmbedding()
    {
        receiverTask.yieldForce();
        dispose(Mallocator.instance, openRequests);
    }

private:

    alias RequestData = Tuple!(size_t[], Slice!(float*, 2, Contiguous));

    void obtainEmbeddingDimension()
    {
        string[1] s = ["test"];
        string[][1] sentences = [s];
        send(sentences[]);
        auto result = receive();
        embeddingDim = result.embeddings.shape[2];
        receiveAllocator.deallocateAll;
    }

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

    void receiveAll(size_t numberOfBatches, bool normalize, void delegate(size_t, size_t) progressCallback)
    {
        foreach (i; 1 .. numberOfBatches + 1)
        {
            receiveEmbeddings(normalize);
            progressCallback(i, numberOfBatches);
        }
    }

    void receiveEmbeddings(bool normalize)
    {
        scope (exit)
            receiveAllocator.deallocateAll;
        with (receive())
        {
            size_t[] sentenceLengths = openRequests[receivedId - idOffset][0];
            Slice!(float*, 2, Contiguous) storage = openRequests[receivedId - idOffset][1];
            immutable seqLen = maxSeqLen ? maxSeqLen : sentenceLengths.maxElement + 2;
            assert(embeddings.shape == [sentenceLengths.length, seqLen, embeddingDim],
                    format!"%s != %s"(embeddings.shape, [sentenceLengths.length, seqLen, embeddingDim]));

            size_t i;
            if (normalize)
            {
                foreach (j; 0 .. sentenceLengths.length)
                    foreach (k; 1 .. sentenceLengths[j] + 1)
                        storage[i++][] = embeddings[j, k] * (1f / nrm2(embeddings[j, k]));
            }
            else
            {
                foreach (j; 0 .. sentenceLengths.length)
                    foreach (k; 1 .. sentenceLengths[j] + 1)
                        storage[i++][] = embeddings[j, k];
            }
            assert(i == storage.length!0);
            dispose(Mallocator.instance, sentenceLengths);
        }
    }

    auto receive()
    {
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
        return tuple!("embeddings", "receivedId")(slice, receivedId);
    }

    __gshared Context context;
    __gshared Socket sender;
    __gshared Socket receiver;
    string uuid;
    size_t requestId;
    __gshared size_t idOffset;
    __gshared RequestData[] openRequests;
    Task!(run, void delegate(size_t, bool, void delegate(size_t, size_t)), size_t, bool, void delegate(size_t, size_t)) receiverTask;
    __gshared int maxSeqLen;
    __gshared bool showTokensToClient;
    __gshared Region!Mallocator receiveAllocator;
    __gshared Region!Mallocator sendAllocator;
    Duration timeout = 60.seconds;
}

unittest
{
    BertEmbedding bert = BertEmbedding();
    bert.timeout = 1.msecs;
    bert.initialize();
    string[][] sentences = [
        ["I", "'m", "not", "sure", "how", "I", "would", "have", "handled", "it", "."],
        ["I", "had", "a", "problem", "with", "the", "tile", "in", "my", "bathroom", "coming", "apart", "."]
    ];
    size_t tokens = sentences.stdMap!(s => s.length)
        .fold!((a, b) => a + b)(0UL);
    auto wordEmbeddings = slice!float(tokens, BertEmbedding.embeddingDim);
    stderr.write("Fetching word embeddings for ", tokens, " tokens...");
    bert.beginEmbedding(1, true, (a, b) => stderr.writeln(a, " ", b));
    bert.embed(sentences, wordEmbeddings);
    bert.endEmbedding();
    stderr.writeln("Done!");
    writeln("Cosine distance ", 1f - cosineSimilarity(wordEmbeddings[3], wordEmbeddings[9]));
}
