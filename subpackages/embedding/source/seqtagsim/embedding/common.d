/**
 * Copyright: Fynn Schröder, 2019
 * Author: Fynn Schröder
 * License: MIT
 */

module seqtagsim.embedding.common;

import mir.ndslice : Slice;
import mir.glas.l1 : dot, nrm2;

float cosineSimilarity(Slice!(float*) a, Slice!(float*) b)
{
    return dot(a, b) / (nrm2(a) * nrm2(b));
}


struct EmbeddingBase
{
    size_t embeddingDim();
    void initialize(string dummy) {}
    void beginEmbedding(size_t numberOfBatches, void function(size_t, size_t) progressCallback) {}
    void normEmbeddings(string[][] sentences, Slice!(float*, 2) storage);
    void endEmbedding() {}
}
