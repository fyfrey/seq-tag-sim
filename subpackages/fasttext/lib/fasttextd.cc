/**
 * Copyright: Fynn Schröder, 2019
 * Author: Fynn Schröder
 * License: MIT
 */

#include "fasttext.h"
#include "string.h"

using namespace fasttext;
using namespace std;

struct Dstring
{
    size_t length;
    const char* ptr;
};

struct Pair
{
    real prob;
    Dstring word;
};

class FastTextWrapper
{
private:
    FastText* f;
    Dstring (*copyString)(void*, const char*, size_t);
    void* dAlloc;
public:
    FastTextWrapper(void* dAlloc, Dstring (*copyString)(void*, const char*, size_t))
    {
        this->copyString = copyString;
        this->dAlloc = dAlloc;
        f = new FastText();
    }

    ~FastTextWrapper()
    {
        delete f;
    }

    void loadModel(const char* filename, size_t length);

    void fillWordVector(const char* word, size_t length, float* vector);

    int getDimension();

    void fillSentenceVector(const char* sentence, size_t length, float* vector);

    void fillAnalogies(int k, const Dstring wordA, const Dstring wordB, const Dstring wordC, Pair* pairs);

    void fillNN(const Dstring word, int k, Pair* pairs);

    void destroy();
};

void FastTextWrapper::loadModel(const char* filename, size_t length)
{
    f->loadModel(string(filename, length));
}

void FastTextWrapper::fillWordVector(const char* word, size_t length, float* vector)
{
    Vector vec = Vector(f->getDimension());
    f->getWordVector(vec, string(word, length));
    memcpy(vector, vec.data(), vec.size() * sizeof(float));
}

int FastTextWrapper::getDimension()
{
    return f->getDimension();
}

struct membuf: std::streambuf {
    membuf(char const* base, size_t size) {
        char* p(const_cast<char*>(base));
        this->setg(p, p, p + size);
    }
};
struct imemstream: virtual membuf, std::istream {
    imemstream(char const* base, size_t size)
        : membuf(base, size)
        , std::istream(static_cast<std::streambuf*>(this)) {
    }
};

void FastTextWrapper::fillSentenceVector(const char* sentence, size_t length, float* vector)
{
    imemstream in(sentence, length);
    Vector vec = Vector(f->getDimension());
    f->getSentenceVector(in, vec);
    memcpy(vector, vec.data(), vec.size() * sizeof(float));
}

void FastTextWrapper::fillAnalogies(int32_t k, const Dstring wordA, const Dstring wordB, const Dstring wordC, Pair* pairs)
{
    std::vector<std::pair<real, std::string>> analogies = f->getAnalogies(k, string(wordA.ptr, wordA.length), string(wordB.ptr, wordB.length), string(wordC.ptr, wordC.length));
    for (size_t i = 0; i < analogies.size(); i++)
    {
        pairs[i] = Pair{analogies[i].first, copyString(dAlloc, analogies[i].second.data(), analogies[i].second.size())};
    }
}

void FastTextWrapper::fillNN(const Dstring word, int k, Pair* pairs)
{
    std::vector<std::pair<real, std::string>> neighbors = f->getNN(string(word.ptr, word.length), k);
    for (size_t i = 0; i < neighbors.size(); i++)
    {
        pairs[i] = Pair{neighbors[i].first, copyString(dAlloc, neighbors[i].second.data(), neighbors[i].second.size())};
    }
}


void FastTextWrapper::destroy()
{
    delete this;
}

FastTextWrapper *createInstance(void* dAlloc, Dstring (*copyString)(void*, const char*, size_t))
{
    return new FastTextWrapper(dAlloc, copyString);
}

