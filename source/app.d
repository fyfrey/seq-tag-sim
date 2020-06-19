/**
 * Copyright: Fynn Schröder, 2019
 * Author: Fynn Schröder
 * License: MIT
 */

import std.stdio;
import std.algorithm;
import std.getopt;

import seqtagsim.compare;

version (unittest)
{
}
else
{
	void main(string[] args)
	{
		processArgs(args);
	}
}

private void processArgs(string[] args)
{
	import std.format : format;
	import std.traits : EnumMembers;
	import std.typecons : tuple;
	import std.parallelism : defaultPoolThreads;
	import seqtagsim.reader : FileFormat;

	GetoptResult result;
	FileFormat[] formats;
	CompareConfig compareConfig;
	with (compareConfig)
	{
		try
		{
			uint threads;
			version (embedding)
				auto extraOptions = tuple("c|context",
						"Compare contextual embeddings of each token: " ~ format!"<%(%s|%)>"([
								EnumMembers!(CompareConfig.Context)
							]), &context, "s|similarity-threshold", "Minimal similarity [0,1] for word embeddings (default 0)", &similarityThreshold,
						"e|embedding-location",
						"path to fastText word vector file (in bin or ftz format with subword information) or BERT service address",
						&embeddings, "split-sentences",
						"Split multi-sentence segments into single sentences for embedding compatibility (default false)",
						&splitSentences,
						"fuse-spans",
						"Fuse same-label spans of multiple tokens to a single token for comparison (default false)",
						&fuseMultiTokenSpans,
						"intermediate-marker", "Character marking a label as an intermediate, continuing label (default I)", &iMarker);
			else
				auto extraOptions = tuple();
			result = getopt(args, config.passThrough, config.keepEndOfOptions, extraOptions.expand, "f|format",
					"File format, use twice when dataset formats differ: " ~ format!"<%(%s|%)>"([
							EnumMembers!(FileFormat)
						]), &formats, "d|details", "Print details for all similarity measures", &allMeasureDetails, "p|pattern",
					"GLOB pattern to select files (mainly useful with limited shell globbing), e.g. *.txt, use twice",
					&patterns, "t|threads",
					"number of threads to use for parallelizable operations (defaults to the number of logical cores)", &threads);
			args = args[1 .. $];
			defaultPoolThreads(threads - 1);
		}
		catch (Exception e)
			return stderr.writeln(e.msg);

		if (result.helpWanted || args.length < 2)
		{
			printHelp(result);
			return;
		}

		ptrdiff_t index;
		if (args.length == 2)
		{
			dataset1Paths = args[0 .. 1];
			dataset2Paths = args[1 .. 2];
		}
		else if (args.length > 2 && (index = countUntil(args, "--")) != -1)
		{
			dataset1Paths = args[0 .. index];
			dataset2Paths = args[index + 1 .. $];
		}
		else
		{
			stderr.writeln("Multiple input files are given, but no -- separator to know which files belong to dataset 1 or 2");
			printHelp(result);
			return;
		}

		if (formats.length == 2)
		{
			fileFormat1 = formats[0];
			fileFormat2 = formats[1];
		}
		else if (formats.length == 1)
		{
			fileFormat1 = formats[0];
			fileFormat2 = formats[0];
		}
		else if (formats.length > 2)
		{
			stderr.writeln("More than two file formats are given, but only one or two are allowed");
			printHelp(result);
			return;
		}

		selectAndPerformComparison(compareConfig);
	}

}

private void printHelp(ref GetoptResult result)
{
	string msg = r"Comparison requires two files/folders
Usage: seq-tag-sim [OPTIONS] dataset1Path dataset2Path
When globbing multiple files per dataset, use -- to separate the two datasets";
	defaultGetoptFormatter(stderr.lockingTextWriter(), msg, result.options);
}

version (python)
{
	private import pyd.pyd;

	shared static this()
	{
		//initializes PyD
		py_init();
	}

	shared static ~this()
	{
		// cleanup PyD
		py_finish();
	}
}
