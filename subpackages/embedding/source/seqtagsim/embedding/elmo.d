/**
 * Copyright: Fynn Schröder, 2019
 * Author: Fynn Schröder
 * License: MIT
 */

module seqtagsim.embedding.elmo;

version (python):
import seqtagsim.embedding.common;
import seqtagsim.util;

import mir.ndslice : Slice, Universal;
import mir.glas.l1 : nrm2;
import pyd.pyd;
import pyd.embedded;
import pyd.extra;

/// Provides access to AllenNLP's ELMo Python library via pyd 
@extends!EmbeddingBase struct ElmoEmbedding
{
    mixin base;

    enum embeddingDim = 1024;

    /// Initializes the embeddings by creating an instance in Python
    void initialize(string dummy)
    {
        python = new InterpContext();
        python.py_stmts("from allennlp.commands.elmo import ElmoEmbedder\nelmo = ElmoEmbedder()\n");
    }

    /// Embeds a single batch
    void embed(string[][] sentences, Slice!(float*, 2) storage)
    {
        python.sentences = sentences;
        PydObject emb = python.py_eval("elmo.embed_batch(sentences)");
        size_t i;
        foreach (PydObject npArray; emb)
        {
            Slice!(float*, 3, Universal) slice = numpyToMir(npArray);
            foreach (j; 0 .. slice.length!1)
            {
                auto vector = storage[i++];
                vector[] = slice[0, j];
                vector[] += slice[1, j];
                vector[] += slice[2, j];
                if (normalize)
                    vector[] *= 1f / nrm2(vector);
            }
        }
    }

private:
    InterpContext python;

    Slice!(float*, 3, Universal) numpyToMir(PydObject numpyArray)
    {
        import mir.ndslice.connect.cpython : fromPythonBuffer, pythonBufferFlags, PythonBufferErrorCode, bufferinfo;
        import deimos.python.abstract_ : PyObject_GetBuffer;
        import deimos.python.object : PyObject, Py_buffer;

        Slice!(float*, 3, Universal) mat = void;
        Py_buffer bufferView;
        immutable flags = pythonBufferFlags!(mat.kind, const(float));
        immutable getBufferResult = PyObject_GetBuffer(numpyArray.tupleof[0], &bufferView, flags);
        assert(getBufferResult == 0);
        bufferinfo bufferInfo = *cast(bufferinfo*)&bufferView;
        immutable result = fromPythonBuffer(mat, bufferInfo);
        assert(PythonBufferErrorCode.success == result);
        return mat;
    }
}

unittest
{
    ElmoEmbedding elmo;
    elmo.initialize();
    string[][] sentences = [
        ["I", "'m", "not", "sure", "how", "I", "would", "have", "handled", "it", "."],
        ["I", "had", "a", "problem", "with", "the", "tile", "in", "my", "bathroom", "coming", "apart", "."]
    ];
    size_t tokens = sentences.map!(s => s.length)
        .fold!((a, b) => a + b)(0UL);
    auto wordEmbeddings = slice!float(tokens, 1024);
    stderr.write("Fetching word embeddings for ", tokens, " tokens...");
    elmo.embed(sentences, wordEmbeddings);
    stderr.writeln("Done!");
    writeln("Cosine distance ", 1f - cosineSimilarity(wordEmbeddings[3], wordEmbeddings[9]));
}
