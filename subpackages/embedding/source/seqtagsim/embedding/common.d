/**
 * Copyright: Fynn Schröder, 2019
 * Author: Fynn Schröder
 * License: MIT
 */

module seqtagsim.embedding.common;

import mir.ndslice : Slice;
import mir.glas.l1 : dot, nrm2;

/// Base type for all embeddings
struct EmbeddingBase
{
    size_t embeddingDim();
    void initialize(string dummy)
    {
    }

    void beginEmbedding(size_t numberOfBatches, bool normalize = true, void delegate(size_t, size_t) progressCallback = null)
    {
        this.normalize = normalize;
    }

    void embed(string[][] sentences, Slice!(float*, 2) storage);
    void endEmbedding()
    {
    }

    bool normalize;
}

version (unittest)
{
    float cosineSimilarity(Slice!(float*) a, Slice!(float*) b)
    {
        return dot(a, b) / (nrm2(a) * nrm2(b));
    }
}
