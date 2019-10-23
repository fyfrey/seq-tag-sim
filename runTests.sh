#!/bin/sh

dub test :blas
dub test :cuda
dub test :embedding
dub test :fasttext
dub test :reader
dub test :util
dub test

dub clean --all-packages
