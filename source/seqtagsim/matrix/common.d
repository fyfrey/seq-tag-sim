/**
 * Copyright: Fynn Schröder, 2019
 * Author: Fynn Schröder
 * License: MIT
 */

module seqtagsim.matrix.common;

private import mir.ndslice;
private import mir.glas.l1;

public import std.typecons;

/**
 * Finds the most similar word vector according to cosine similarity.
 *
 * Params:
 *     otherEmbeddings = Matrix with any number of normalized word embeddings
 *     embedding = The normalized word embedding to compare against the other embeddings
 *
 * Returns:
 *     Tuple of maximal similarity and its index
 */
Tuple!(float, uint) computeSimilarity(scope Slice!(const(float)*, 2) otherEmbeddings, scope Slice!(const(float)*) embedding) pure nothrow
{
    return reduce!((a, b, c) => b > a[0] ? tuple(b, c) : a)(tuple(-float.max, 0U), otherEmbeddings.byDim!0
            .map!(e => dot(e, embedding)), iota!uint(otherEmbeddings.shape[0]));
}

unittest
{
    Slice!(float*, 2) matrix = slice!float(1000, 100);
    matrix[] = 0.5;
    matrix[900][] = 1;
    Slice!(float*, 1) vector = slice!float(100);
    vector[] = 0.2;
    Tuple!(float, uint) result = computeSimilarity(matrix, vector);
    assert(result[1] == 900);
}
