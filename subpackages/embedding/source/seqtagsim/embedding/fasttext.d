/**
 * Copyright: Fynn Schröder, 2019
 * Author: Fynn Schröder
 * License: MIT
 */

module seqtagsim.embedding.fasttext;

version (fasttext):

import seqtagsim.embedding.common;
import seqtagsim.util;
import seqtagsim.fasttext;

import mir.ndslice.slice;
import mir.glas.l1;

import std.experimental.allocator;
import std.experimental.allocator.building_blocks.region;
import std.experimental.allocator.mallocator;

/// Provides fastText embeddings using the custom wrapper 
@extends!EmbeddingBase struct FastTextEmbedding
{
    mixin base;

    /// Loads the fastText model from file
    void initialize(string modelFile)
    {
        alloc = Region!Mallocator(1024 * 1024);
        ft = FastText!(Region!Mallocator)(alloc);
        ft.loadModel(modelFile);
    }

    /// Returns the dimensions of a single embedding
    size_t embeddingDim()
    {
        return ft.getDimension();
    }

    /// Embeds a single batch
    void embed(string[][] sentences, Slice!(float*, 2) storage)
    {
        size_t i;
        foreach (string[] sentence; sentences)
            foreach (string word; sentence)
            {
                auto vector = storage[i++];
                ft.fillWordVector(word, vector);
                if (normalize)
                    vector[] *= 1f / nrm2(vector);
            }
    }

private:
    FastText!(Region!Mallocator) ft;
    Region!Mallocator alloc;
}
