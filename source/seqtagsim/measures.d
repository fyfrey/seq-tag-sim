/**
 * Copyright: Fynn Schröder, 2019
 * Author: Fynn Schröder
 * License: MIT
 */

module seqtagsim.measures;

private import mir.ndslice;
private import mir.ndslice.allocation;
private import mir.math.common;
private import mir.math.sum;
private import mir.rc;

pure nothrow @nogc:

/**
 * Structure containing various information-theoretic measures.
 */
struct InformationTheoreticMeasures
{
    double entropyRow;
    double entropyCol;
    double jointEntropy;

    double mutualInformation;
    double conditionalEntropyRow;
    double conditionalEntropyCol;
    double variationOfInformation;

    double normalizedVariationOfInformation;
    double normalizedMutualInformationSum;
    double normalizedMutualInformationJoint;
    double normalizedMutualInformationMax;

    double homogeneity;
    double completeness;
    double vMeasure;
}

/**
 * Computes information-theoretic clustering comparison measures on the given counts from a contingency table.
 *
 * Params:
 *     matrix = Contigency table with counts, but without the sum column or row
 *
 * Returns:
 *     Information-theoretic measure values
 */
InformationTheoreticMeasures computeInformationTheoreticMeasuresFromMatrix(const scope Slice!(double*, 2) matrix)
{
    Slice!(RCI!double) thisN = rcslice(matrix.lightConst
            .byDim!0
            .map!sum);
    Slice!(RCI!double) otherN = rcslice(matrix.lightConst
            .byDim!1
            .map!sum);
    immutable double totalN = sum(thisN);
    assert(approxEqual(totalN, sum(otherN)));

    // frequencies as probability estimates
    Slice!(RCI!double) thisP = rcslice(thisN / totalN);
    Slice!(RCI!double) otherP = rcslice(otherN / totalN);
    Slice!(RCI!double, 2) jointP = rcslice(matrix / totalN);

    InformationTheoreticMeasures result;
    with (result)
    {
        entropyRow = -sum(thisP * map!log2(thisP));
        entropyCol = -sum(otherP * map!log2(otherP));
        jointEntropy = -sum(jointP * map!log2(jointP));
        mutualInformation = sum(jointP * map!log2(jointP / cartesian(thisP, otherP).map!"a * b"));

        conditionalEntropyRow = entropyRow - mutualInformation;
        conditionalEntropyCol = entropyCol - mutualInformation;
        variationOfInformation = entropyRow + entropyCol - 2 * mutualInformation;
        normalizedVariationOfInformation = entropyRow > 0.0 ? variationOfInformation / entropyRow : entropyCol;

        normalizedMutualInformationSum = 2.0 * mutualInformation / (entropyRow + entropyCol);
        normalizedMutualInformationJoint = mutualInformation / jointEntropy;
        normalizedMutualInformationMax = mutualInformation / fmax(entropyRow, entropyCol);

        homogeneity = 1 - (conditionalEntropyRow / entropyRow);
        completeness = 1 - (conditionalEntropyCol / entropyCol);
        vMeasure = 2 * homogeneity * completeness / (homogeneity + completeness);
        assert(approxEqual(vMeasure, normalizedMutualInformationSum));
    }
    return result;
}

unittest
{
    auto matrix = rcslice!double(4, 4);
    matrix[] = double.epsilon;
    matrix[0, 0] = 3;
    matrix[1, 0] = 2;
    matrix[2, 0] = 2;
    matrix[3, 0] = 3;
    matrix[3, 1] = 2;
    matrix[3, 2] = 2;
    matrix[3, 3] = 2;

    immutable InformationTheoreticMeasures itm = computeInformationTheoreticMeasuresFromMatrix(matrix.lightScope);
    assert(approxEqual(itm.normalizedMutualInformationSum, 0.272107));
    assert(approxEqual(itm.normalizedMutualInformationJoint, 0.157479));
    assert(approxEqual(itm.normalizedMutualInformationMax, 0.262252));
}

unittest
{
    auto matrix = rcslice!double(4, 4);
    matrix[] = double.epsilon;
    matrix[0, 0] = 17;
    matrix[1, 3] = 12;
    matrix[2, 1] = 19;
    matrix[3, 2] = 12;

    immutable InformationTheoreticMeasures itm = computeInformationTheoreticMeasuresFromMatrix(matrix.lightScope);
    assert(approxEqual(itm.normalizedMutualInformationSum, 1.0));
    assert(approxEqual(itm.normalizedMutualInformationJoint, 1.0));
    assert(approxEqual(itm.normalizedMutualInformationMax, 1.0));
}

unittest
{
    auto matrix = rcslice!double(4, 4);
    matrix[] = double.epsilon;
    matrix[0, 0] = 10;
    matrix[1, 1] = 11;
    matrix[2, 2] = 12;
    matrix[3, 3] = 13;
    matrix[0, 3] = 1;
    matrix[1, 0] = 1;
    matrix[2, 1] = 2;
    matrix[3, 2] = 3;

    immutable InformationTheoreticMeasures itm = computeInformationTheoreticMeasuresFromMatrix(matrix.lightScope);
    assert(approxEqual(itm.normalizedMutualInformationSum, 0.724106));
    assert(approxEqual(itm.normalizedMutualInformationJoint, 0.567528));
    assert(approxEqual(itm.normalizedMutualInformationMax, 0.723041));
}

unittest
{
    auto matrix = rcslice!double(4, 4);
    matrix[] = double.epsilon;
    matrix[0, 0] = 15;
    matrix[1 .. 4, 1 .. 4] = 5;

    immutable InformationTheoreticMeasures itm = computeInformationTheoreticMeasuresFromMatrix(matrix.lightScope);
    assert(approxEqual(itm.normalizedMutualInformationSum, 0.405639));
    assert(approxEqual(itm.normalizedMutualInformationJoint, 0.254421));
    assert(approxEqual(itm.normalizedMutualInformationMax, 0.405639));
}

unittest
{
    auto matrix = rcslice!double(4, 10);
    matrix[] = double.epsilon;
    matrix[0, 0] = 10;
    matrix[0, 3] = 20;
    matrix[0, 9] = 2;
    matrix[1, 1] = 12;
    matrix[1, 2] = 12;
    matrix[1, 8] = 1;
    matrix[2, 3] = 4;
    matrix[2, 4] = 2;
    matrix[2, 5] = 15;
    matrix[2, 7] = 5;
    matrix[3, 0] = 1;
    matrix[3, 4] = 1;
    matrix[3, 6] = 10;
    matrix[3, 8] = 14;

    immutable InformationTheoreticMeasures itm = computeInformationTheoreticMeasuresFromMatrix(matrix.lightScope);
    assert(approxEqual(itm.normalizedMutualInformationSum, 0.68375));
    assert(approxEqual(itm.normalizedMutualInformationJoint, 0.519468));
    assert(approxEqual(itm.normalizedMutualInformationMax, 0.563671));
}
