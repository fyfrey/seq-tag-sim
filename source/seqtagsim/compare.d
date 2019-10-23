/**
 * Copyright: Fynn Schröder, 2019
 * Author: Fynn Schröder
 * License: MIT
 */

module seqtagsim.compare;

import std.stdio;
import std.algorithm;
import std.datetime.stopwatch : StopWatch, AutoStart;
import std.experimental.allocator.mallocator : Mallocator;
import std.typecons : scoped;
import std.parallelism : taskPool, totalCPUs, scopedTask, defaultPoolThreads;
import std.meta;
import std.range;

import seqtagsim.util;
import seqtagsim.reader;
import seqtagsim.matrix.token;

import seqtagsim.matrix;
import seqtagsim.measures;

alias Alloc = typeof(Mallocator.instance);

/**
 * Configuration Options for the dataset comparison.
 */
struct CompareConfig
{
	string[] patterns;
	string[] dataset1Paths;
	string[] dataset2Paths;
	FileFormat fileFormat1 = FileFormat.deduce;
	FileFormat fileFormat2 = FileFormat.deduce;

	version (embedding)
	{
		string embeddings;
		float similarityThreshold = 0.0f;
		Context context = Context.none;
		enum Context
		{
			none,
			bert,
			elmo
		}
	}
}

/**
 * Selects and performs the comparison according to the given configuration.
 *
 * Params:
 *     config = Configuration
 */
void selectAndPerformComparison(const ref CompareConfig config)
{
	version (embedding)
	{
		import seqtagsim.embedding;

		if (config.context != CompareConfig.Context.none)
		{
			if (config.context == CompareConfig.Context.elmo)
			{
				version (python)
					return compare!(Dataset!ElmoEmbedding, ElmoEmbedding)(config);
				else
					return stderr.writeln("Cannot use ELMo embeddings because this program is not compiled with Python support!");
			}
			else if (config.context == CompareConfig.Context.bert)
				return compare!(Dataset!BertEmbedding, BertEmbedding)(config);
		}
		else
		{
			version (fasttext)
				if (config.embeddings != null)
					return compare!(EmbeddingTextOverlap!FastTextEmbedding, FastTextEmbedding)(config);
		}
	}
	compare!(Vocabulary, void)(config);
}

private:

void compare(Type, Embedding)(const ref CompareConfig config)
{
	StopWatch sw = StopWatch(AutoStart.yes);

	static if (is(Embedding == void))
	{
		Type d1;
		Type d2;
	}
	else
	{
		Embedding emb;
		stderr.write("Initializing embedding...");
		auto loadModel = scopedTask({ emb.initialize(config.embeddings); });
		taskPool.put(loadModel);

		sw.reset();
		auto d1 = Type!Embedding(emb);
		auto d2 = Type!Embedding(emb);
	}

	auto files1 = config.dataset1Paths.length > 1 ? config.dataset1Paths : listFiles(config.dataset1Paths[0],
			config.patterns.length ? config.patterns[0] : null).array;
	auto files2 = config.dataset2Paths.length > 1 ? config.dataset2Paths : listFiles(config.dataset2Paths[0],
			config.patterns.length ? config.patterns[1] : null).array;
	auto task1 = scopedTask({ files1.each!(f => processByFilename!(d1.read)(f, config.fileFormat1, d1)); d1.endReading(); });
	auto task2 = scopedTask({ files2.each!(f => processByFilename!(d2.read)(f, config.fileFormat2, d2)); d2.endReading(); });
	taskPool.put(task1);
	taskPool.put(task2);

	static if (!is(Embedding == void))
	{
		loadModel.yieldForce();
		stderr.writefln!"done! It took %s ms"(sw.peek.total!"msecs");
		sw.reset();
	}
	task1.yieldForce();
	d1.beginEmbedding();
	stderr.writefln!"Preparing dataset 1 took %s ms"(sw.peek.total!"msecs");
	sw.reset();
	task2.yieldForce();
	d2.beginEmbedding();
	stderr.writefln!"Preparing dataset 2 took %s ms"(sw.peek.total!"msecs");
	sw.reset();
	writeln("Comparing dataset A against dataset B");
	auto result = d1.compare(d2);
	writeln("TagsA: ", result[0].length);
	writeln("TagsB: ", result[0].length!1);
	foreach (i, name; result.fieldNames)
	{
		writeln("\nResults for ", name, ":");
		computeInformationTheoreticMeasuresFromMatrix(result[i].lightScope).prettyPrintStruct;
	}
	stderr.writefln!"\nComparing datasets took %s ms"(sw.peek.total!"msecs");
}

void processByFilename(alias method, T)(string filename, FileFormat format, ref T processor)
{
	if (format == FileFormat.deduce)
	{
		foreach (Reader; Readers)
			if (filename.endsWith(Reader.fileType))
				return processFile!(Reader, T, method)(filename, processor);
	}
	else
	{
		foreach (Reader; Readers)
			if (format == Reader.fileFormat)
				return processFile!(Reader, T, method)(filename, processor);
	}
}

void processFile(Reader, T, alias method)(string filename, ref T processor)
{
	import std.mmfile : MmFile;

	try
	{
		auto mmf = scoped!MmFile(filename);
		string input = cast(string)(cast(ubyte[]) mmf[]);
		Reader reader = Reader(input);
		mixin("processor." ~ __traits(identifier, method) ~ "(reader);");
	}
	catch (Exception e)
		stderr.writeln("Error in file ", filename, "\n", e);
}
