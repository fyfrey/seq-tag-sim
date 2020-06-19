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
	bool allMeasureDetails;

	version (embedding)
	{
		string embeddings;
		bool splitSentences = false;
		float similarityThreshold = 0.0f;
		bool fuseMultiTokenSpans = false;
		char iMarker = 'I';
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
			immutable DatasetConfig dc = DatasetConfig(config.splitSentences, config.similarityThreshold,
					config.fuseMultiTokenSpans, config.iMarker);
			if (config.context == CompareConfig.Context.elmo)
			{
				version (python)
					return compare!(Dataset!ElmoEmbedding, ElmoEmbedding)(config, dc);
				else
					return stderr.writeln("Cannot use ELMo embeddings because this program is not compiled with Python support!");
			}
			else if (config.context == CompareConfig.Context.bert)
				return compare!(Dataset!BertEmbedding, BertEmbedding)(config, dc);
		}
		else
		{
			version (fasttext)
				if (config.embeddings != null)
					return compare!(EmbeddingTextOverlap!FastTextEmbedding, FastTextEmbedding)(config, config.similarityThreshold);
		}
	}
	compare!(Vocabulary, void, int)(config, 0);
}

private:

void compare(Type, Embedding, Options)(const ref CompareConfig config, const Options options)
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
		auto d1 = Type(emb, options);
		auto d2 = Type(emb, options);
	}

	auto files1 = config.dataset1Paths.length > 1 ? config.dataset1Paths : listFiles(config.dataset1Paths[0],
			config.patterns.length ? config.patterns[0] : null).array;
	auto files2 = config.dataset2Paths.length > 1 ? config.dataset2Paths : listFiles(config.dataset2Paths[0],
			config.patterns.length ? config.patterns[1] : null).array;
	auto task1 = scopedTask({
		files1.each!(f => processByFilename!(d1.read)(f, config.fileFormat1, d1));
		d1.endReading();
	});
	auto task2 = scopedTask({
		files2.each!(f => processByFilename!(d2.read)(f, config.fileFormat2, d2));
		d2.endReading();
	});
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
	auto result = d1.compare(d2);
	writeln("Tags A: ", result[0].length);
	writeln("Tags B: ", result[0].length!1);

	static if (is(typeof(result[$ - 1]) == double))
		printTextOverlapScores(config.allMeasureDetails, result);
	else
		printUnifiedUndirectionalEmbeddingScores(config.allMeasureDetails, config.fuseMultiTokenSpans, result);

	stderr.writefln!"\nComparing datasets took %s ms"(sw.peek.total!"msecs");
	stderr.flush();
	stdout.flush();
}

void printTextOverlapScores(R)(bool showDetails, ref R result)
{
	if (showDetails)
	{
		foreach (i, name; result.fieldNames[0 .. $ - 2])
		{
			writeln("\nResults for method ", name, ":");
			computeInformationTheoreticMeasuresFromMatrix(result[i].lightScope).prettyPrintStruct;
		}
		writeln();
	}
	writefln("Shared tokens: %.3f", result.sharedTokens);
	writefln("Shared vocabulary: %.3f", result.sharedVocabulary);
	const double nmiJointMul = computeInformationTheoreticMeasuresFromMatrix(result.mulIwf.lightScope).normalizedMutualInformationJoint;
	const double nmiJointAdd = computeInformationTheoreticMeasuresFromMatrix(result.addIwf.lightScope).normalizedMutualInformationJoint;
	writefln("NMI Joint (multiplicative): %.3f", nmiJointMul);
	writefln("NMI Joint (additive): %.3f", nmiJointAdd);

	const double textOverlapMul = (2.0 * nmiJointMul * result.sharedVocabulary) / (nmiJointMul + result.sharedVocabulary);
	const double textOverlapAdd = (2.0 * nmiJointAdd * result.sharedVocabulary) / (nmiJointAdd + result.sharedVocabulary);
	writefln("Text Overlap (multiplicative): %.3f", textOverlapMul);
	writefln("Text Overlap (additive): %.3f", textOverlapAdd);
}

void printUnifiedUndirectionalEmbeddingScores(R)(bool showDetails, bool fused, ref R result)
{
	if (showDetails)
	{
		if (fused)
			foreach (i, name; result.fieldNames[$ / 2 .. $])
			{
				writeln("\nResults for method ", name, ":");
				computeInformationTheoreticMeasuresFromMatrix(result[$ / 2 + i].lightScope).prettyPrintStruct;
			}
		else
			foreach (i, name; result.fieldNames[0 .. $ / 2])
			{
				writeln("\nResults for method ", name, ":");
				computeInformationTheoreticMeasuresFromMatrix(result[i].lightScope).prettyPrintStruct;
			}
		writeln();
	}

	double nmiJointAB, nmiJointBA;
	if (fused)
	{
		nmiJointAB = computeInformationTheoreticMeasuresFromMatrix(result.fusedWeightedAB.lightScope).normalizedMutualInformationJoint;
		nmiJointBA = computeInformationTheoreticMeasuresFromMatrix(result.fusedWeightedBA.lightScope).normalizedMutualInformationJoint;
	}
	else
	{
		nmiJointAB = computeInformationTheoreticMeasuresFromMatrix(result.weightedAB.lightScope).normalizedMutualInformationJoint;
		nmiJointBA = computeInformationTheoreticMeasuresFromMatrix(result.weightedBA.lightScope).normalizedMutualInformationJoint;
	}
	const double unifiedUndirectionalEmbeddingScore = (2.0 * nmiJointAB * nmiJointBA) / (nmiJointAB + nmiJointBA);
	writefln("UUE: %.3f", unifiedUndirectionalEmbeddingScore);
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
