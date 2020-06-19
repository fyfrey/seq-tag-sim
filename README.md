# Sequence Tagging Similarity Tool

This program computes the similarity of two annotated sequence tagging datasets based on the contained words and their labels.
The designated use case is to ease and speed up the tedious process of selecting suitable auxiliary training data for neural networks using multi-task learning to augment the primary training with auxiliary data.
Knowing the similarity between the training dataset and different auxiliary datasets quickly allows selecting the most similar dataset, which should also provide the most improvement of the neural network's performance on the main task.

There are no restrictions on the tagsets used in the datasets.
Arbitrary sequence tagging tasks / datasets can be compared.
As of now, the similarity computation has primarily been tested on tasks where each token is tagged individually, e.g. part-of-speech (POS) tagging, or the grouped tokens are short, e.g. named entity recognition (NER).

## Installation

Portable, stand-alone binary builds are available for download on the GitHub release page.
Extract the archive and copy the `seq-tag-sim` file into a directory on your `PATH`.
Alternatively, call the program via its absolute or relative path.

## Usage

Run `seq-tag-sim -h` to print the commandline help.
The general usage is straightforward.
Run `seq-tag-sim path/to/dataset1 path/to/dataset2` to compare dataset 1 with dataset 2 and compute various similarity measures, which are written to the standard output stream.
In case the automatic data format selection (based on filename extensions) fails, use the `-f` option once or twice to manually select the input format.
If your datasets are split across multiple files, use shell glob operations to select the files.
It is now necessary to distinguish both datasets by placing an `--` in between the two datasets.
The example `seq-tag-sim -f bncPOS -f ptbPOS path/to/dataset1/*.xml -- path/to/dataset2/*.pos` shows how to compare multiple XML files from the British National Corpus with some files in the Penn Treebank POS tagging format. 

## Citation

If you use the software for your research, please cite our paper describing the developed methods:
```
@inproceedings{schröder-biemann-2020-estimating,
    title = "Estimating the influence of auxiliary tasks for multi-task learning of sequence tagging tasks",
    author = "Schröder, Fynn and Biemann, Chris",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    year = "2020",
    publisher = "Association for Computational Linguistics"
}
```

## Advanced installation and usage

Optional, advanced features are to use word embeddings to improve the quality of the similarity calculation.
To use advanced features, additional software and data may be required.
Depending on the type of embedding to be used
* download a [fastText](https://fasttext.cc/) model
* install [AllenNLP](https://github.com/allenai/allennlp) in your active Python environment to use contextual [ELMo](https://allennlp.org/elmo) embeddings
* install [bert-as-service](https://github.com/hanxiao/bert-as-service) in your active Python environment, download a suitable model and start the service to use contextual [BERT](https://github.com/google-research/bert) embeddings.

To use non-contextual word embeddings, i.e. fastText, supply the `-e path/to/embedding.bin` option when running the program.
As the fastText library takes some time to load the model, this may add considerable run time overhead when comparing small datasets.
The preferred option, is to use BERT embeddings.
To do so, run `seq-tag-sim -c bert`.
If the `bert-as-serice` server is not running on the same computer, use the `-e` option to set the server's network address.

## Functioning principle

The overlapping vocabulary between the two datasets builds the bridge to compare the corresponding labels of these words.
Without contextual embeddings, the general workflow is the following:
1. Read a dataset and count for each unique word, how often it is tagged with each label
2. Match and compare words of both datasets
a) If a word from the fist dataset is not contained in the second dataset and fastText embeddings are used, the most similar word in second dataset according to the word vectors' cosine similarity is chosen.
b) The counts how often a word has a certain label are combined from both datasets by increasing the counts at the label-pair's position in a global contingency table. In total, there are eight slightly different methods to combine the label counts.
3. Once all words are processed, the contingency table with the label counts acts as a probabilistic mapping between both tagsets. For example, the counts for the tag `NOUN` from dataset 1 may correspond to 85% to `NN` from dataset 2. The remaining 15% could be distributed in roughly equal parts over other labels from dataset 2. Based on this label count contingency table, multiple information theoretic measures are calculated.

The information theoretic measures include e.g. entropy, cross-entropy, mutual information, variation of information and multiple variants of normalized mutual information.
They represent the similarity of the two input datasets.
When contextual embeddings (BERT or ELMo) are active, individual tokens are matched and their the counts at their labels' position is increased.
The matching of tokens works by computing all most similar vector pairs.

## Implementation overview

The source code is structured into the main application and independently usable subpackages.
The main functionality is in the `source` folder with `app.d` defining the entry point.
In subfolders are the implementations of the vocabulary overlap approach (in `word.d`), the token-based approach using contextual embeddings (in `token.d`) and the information theoretic measures (in `measures.d`).
The top-level folder `subpackages` contains various additional functionalities.
Of these subpackages, only `reader` and `util` are essential.
File readers for various common sequence tagging file formats can be found in the `reader` subpackage.
As its name suggests, the `util` subpackage contains utility functions and structures.
The remaining subpackages are all related to the option word embeddings.
The `blas` subpackage contains an efficient functionality to compute the most similar vector pairs between two huge arrays of vectors. It uses a batched matrix multiplication implementation, which can efficiently multiply matrices that do not fit into memory. Along with the computation of these batches, the maximal similar vectors are found.
An API-wise identical implementation for CUDA exists in the `cuda` subpackage. It can optionally divide the computation up across multiple GPUs, which decreases the run time for large datasets of millions of tokens or more.
The `embedding` subpackage contains structures and functions to use the three different embeddings libraries resp. services with a uniform API.
The `fasttext` subpackage is home to the external fasttext source code and some custom wrapper code to make the usage as a library instead of commandline program possible.

## Building from source

Builder the program from source should be possible on any most current POSIX-like systems (e.g. Linux, FreeBSD, MacOS) and Windows.
To build the software from source, first clone this repository.
A [D language](https://dlang.org) compiler needs to be installed, e.g. [DMD](https://dlang.org/download.html#dmd) (tested with version 2.091.1) or [LDC](https://github.com/ldc-developers/ldc#installation) (tested with version 1.21.0 and 1.22.0).
If the D compiler installation does not include [DUB](https://dub.pm/getting_started) (the D package manager), downloading and installing DUB separately is necessary.
Further, the system's default compiler C/C++ compiler (e.g. gcc or clang) and linker has to be installed.
Building the basic version of the program without support for word embeddings is straightforward: Run `dub build -b release` to produce the `seq-tag-sim` binary.

To build with all word embeddings, additional steps are required.
Run `git submodule update --init --recursive` to get the referenced fastText library sources.
In addition, Python and the development version of the [ZeroMQ](https://zeromq.org/) library `libzmq` needs to be installed on the build system.  
Next, run `dub build -c embedding -b release` to the produce the runnable binary.
Note that the use of contextual embeddings greatly increases the run time as a naïve approach of word vector comparison is used.
To mitigate this problem, additional libraries are required.
If the system has a CUDA-capable GPU, it can be leveraged to speed up the similarity computation process by an order of magnitude.
This requires the [NVIDAI CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) (in version 10.1) to be installed and configured correctly.
Run `dub build -c cuda -b release` to build an optimized version using CUDA for word vector operation acceleration.
If CUDA cannot be used, installation of the [Intel Math Kernel Library (MKL)](https://software.intel.com/en-us/mkl) is recommended.
After sourcing the environment variables by running `~/intel/bin/compilervars.sh intel64`, compiling the software with MKL can be done with `dub build -c blas -b release`.

## Unit tests

To run the main unit tests, call `dub test`.
This does not include the tests of the subpackages.
You can run these individually by calling `dub test :subPackageName` like `dub test :util`.
To run all unit tests invoke `runTests.sh`. Some tests require a CUDA environment and running `bert-as-service` server. 

## Contributing

Contributions are welcome! Raise an issue if you encounter problems or have enhancement proposals.
In the best case, open a pull request with your improvements.