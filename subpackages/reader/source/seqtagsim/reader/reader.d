/**
 * Copyright: Fynn Schröder, 2019
 * Author: Fynn Schröder
 * License: MIT
 */

module seqtagsim.reader.reader;

private import std.typecons : Tuple;
private import std.algorithm : joiner;
private import std.meta : AliasSeq;
private import seqtagsim.util;

alias Pair = Tuple!(string, "word", string, "tag");

immutable string[][string] posTagSets;

shared static this()
{
    posTagSets = ["bnc" : bncC5Tags, "ud" : udPosTags, "ptb" : ptbPosTags];
}

version (allparser)
    alias Readers = AliasSeq!(BinaryReader, TabNewlineReader, UdPosReader,
            GermanNerReader, WnutConllReader, OntonotesConllPosReader,
            OntonotesConllNerReader, ConllNerReader, ConllGerNerReader,
            WikiGoldNerReader, GmbNerReader, BncPosReader, PtbPosReader,
            SecNerReader, EpConllNerReader);
else
    alias Readers = AliasSeq!(BinaryReader, TabNewlineReader, UdPosReader,
            GermanNerReader, WnutConllReader, OntonotesConllPosReader,
            OntonotesConllNerReader, ConllNerReader, ConllGerNerReader,
            WikiGoldNerReader, GmbNerReader, SecNerReader, EpConllNerReader);

enum FileFormat
{
    deduce,
    bin,
    tab,
    udPOS,
    germanerNER,
    wnutConll,
    ontoConllPOS,
    ontoConllNER,
    conllNER,
    conllGerNER,
    wikigoldNER,
    gmbPOS,
    gmbNER,
    bncPOS,
    ptbPOS,
    secNER,
    epConllNER
}

struct TabNewlineReader
{
    import std.algorithm : splitter;
    import std.utf : byCodeUnit;

@safe @nogc nothrow:

    static immutable fileType = ".seq";
    enum fileFormat = FileFormat.tab;

    @disable this();

    this(string content)
    {
        this.text = content;
        this.pairs = joiner(bySentence());
    }

    Pair front()
    {
        return pairs.front;
    }

    void popFront()
    {
        pairs.popFront();
    }

    bool empty()
    {
        return pairs.empty;
    }

    SentenceRange bySentence()
    {
        return SentenceRange(text);
    }

    alias bySegment = bySentence;

    struct SentenceRange
    {
    @safe @nogc nothrow:

        WordTagRange front()
        {
            return WordTagRange(lines.front);
        }

        void popFront()
        {
            lines.popFront();
        }

        bool empty()
        {
            return lines.empty();
        }

        SentenceRange save()
        {
            return this;
        }

    private:
        Splitter lines;

        this(string text)
        {
            lines = text.byCodeUnit.splitter('\n');
        }
    }

    struct WordTagRange
    {
    @safe @nogc nothrow:

        bool empty;

        Pair front()
        {
            return pair;
        }

        void popFront()
        {
            if (sentence.empty)
            {
                empty = true;
                return;
            }
            pair.word = sentence.front.source;
            sentence.popFront();
            pair.tag = sentence.front.source;
            sentence.popFront();
        }

    private:
        Splitter sentence;
        Pair pair;

        this(typeof(byCodeUnit("")) line)
        {
            this.sentence = splitter(line, '\t');
            popFront();
        }
    }

private:
    alias Splitter = typeof(splitter("".byCodeUnit, '\n'));
    typeof(joiner(bySentence())) pairs;
    string text;
}

struct BinaryReader
{
    import std.exception : assumeUnique;
    import std.string : assumeUTF;
    import std.bitmanip : peek, read, Endian;

    static immutable fileType = ".bin";
    enum fileFormat = FileFormat.bin;

    this(string content)
    {
        this(cast(ubyte[]) content);
    }

@safe @nogc nothrow:

    this(ubyte[] content)
    {
        this.content = content;
        initTags();
        this.pairs = joiner(bySentence());
    }

    ~this() @trusted
    {
        import std.experimental.allocator.mallocator : Mallocator;
        import std.experimental.allocator : dispose;
        if (--(*refCount) == 0)
            Mallocator.instance.dispose(tags);
    }

    this(this)
    {
        (*refCount)++;
    }

    Pair front()
    {
        return pairs.front;
    }

    void popFront()
    {
        pairs.popFront();
    }

    bool empty()
    {
        return pairs.empty;
    }

    SentenceRange bySentence()
    {
        return SentenceRange(tags, content[offset .. $]);
    }

    alias bySegment = bySentence;

    struct SentenceRange
    {
    @safe @nogc nothrow:

        WordTagRange front()
        {
            return WordTagRange(tags, content[offset + 2 .. $]);
        }

        void popFront()
        {
            empty = ++current == length;
            if (empty)
                return;
            immutable bytesInSentence = peek!(ushort, Endian.littleEndian)(content[offset .. offset + 2]);
            offset += bytesInSentence;
        }

        bool empty;

        SentenceRange save()
        {
            return this;
        }

    private:

        uint current;
        uint length;
        ubyte[] content;
        string[] tags;
        size_t offset;

        this(string[] tags, ubyte[] content)
        {
            this.tags = tags;
            this.content = content;
            length = peek!(uint, Endian.littleEndian)(content[offset .. offset += 4]);
            empty = current == length;
        }
    }

    struct WordTagRange
    {
    @safe @nogc nothrow:

        Pair front()
        {
            return pair;
        }

        void popFront() @trusted
        {
            empty = current++ == wordCount;
            if (empty)
                return;
            immutable uint tagId = content[offset++];
            uint wordLength = content[offset++];
            if (wordLength == ubyte.max)
                wordLength = peek!(ushort, Endian.littleEndian)(content[offset .. offset += 2]);
            pair.word = cast(string)(content[offset .. offset += wordLength]);
            pair.tag = tags[tagId];
        }

        bool empty = true;

    private:
        string[] tags;
        ubyte[] content;
        uint wordCount;
        uint current;
        size_t offset;
        Pair pair;

        this(string[] tags, ubyte[] content) @trusted
        {
            this.tags = tags;
            this.content = content;
            wordCount = peek!(ushort, Endian.littleEndian)(content[offset .. offset += 2]);
            if (wordCount > 0)
                popFront();
        }
    }

private:
    typeof(joiner(bySentence())) pairs;
    ubyte[] content;
    size_t offset;
    string[] tags;
    uint* refCount;

    void initTags() @trusted
    {
        import std.experimental.allocator.mallocator : Mallocator;
        import std.experimental.allocator : make, makeArray;

        refCount = Mallocator.instance.make!uint(1);
        immutable size_t numberOfTags = content[offset++];
        tags = Mallocator.instance.makeArray!string(numberOfTags);
        foreach (ref string tag; tags)
        {
            size_t l = content[offset++];
            tag = cast(string)(content[offset .. offset += l]);
        }
    }
}

static immutable string[] bncPosTags = [
    "ADJ", "ADV", "ART", "CONJ", "INTERJ", "PREP", "PRON", "PUL", "PUN", "PUQ", "PUR", "SUBST", "UNC", "VERB"
];

static immutable string[] bncC5Tags = [
    "AJ0", "AJ0-AV0", "AJ0-NN1", "AJ0-VVD", "AJ0-VVG", "AJ0-VVN", "AJC", "AJS", "AT0", "AV0",
    "AV0-AJ0", "AVP", "AVP-PRP", "AVQ", "AVQ-CJS", "CJC", "CJS", "CJS-AVQ", "CJS-PRP", "CJT",
    "CJT-DT0", "CRD", "CRD-PNI", "DPS", "DT0", "DT0-CJT", "DTQ", "EX0", "ITJ", "NN0", "NN1",
    "NN1-AJ0", "NN1-NP0", "NN1-VVB", "NN1-VVG", "NN2", "NN2-VVZ", "NP0", "NP0-NN1", "ORD", "PNI",
    "PNI-CRD", "PNP", "PNQ", "PNX", "POS", "PRF", "PRP", "PRP-AVP", "PRP-CJS", "PUL", "PUN", "PUQ",
    "PUR", "TO0", "UNC", "VBB", "VBD", "VBG", "VBI", "VBN", "VBZ", "VDB", "VDD", "VDG", "VDI",
    "VDN", "VDZ", "VHB", "VHD", "VHG", "VHI", "VHN", "VHZ", "VM0", "VVB", "VVB-NN1", "VVD",
    "VVD-AJ0", "VVD-VVN", "VVG", "VVG-AJ0", "VVG-NN1", "VVI", "VVN", "VVN-AJ0", "VVN-VVD", "VVZ",
    "VVZ-NN2", "XX0", "ZZ0"
];

version(allparser)
{

struct BncPosReader
{
    import dxml.parser : EntityRange, simpleXML, parseXML, EntityType;

    static immutable fileType = ".xml";
    enum fileFormat = FileFormat.bncPOS;
    static immutable tagSet = bncC5Tags;

    this(string content)
    {
        parse = parseXML!simpleXML(content);
        pairs = joiner(bySegment());
    }

    Pair front()
    {
        return pairs.front;
    }

    void popFront()
    {
        pairs.popFront();
    }

    bool empty()
    {
        return pairs.empty;
    }

    SegmentRange bySegment()
    {
        return SegmentRange(parse);
    }

    struct SegmentRange
    {
        WordTagRange front()
        {
            return WordTagRange(range);
        }

        void popFront()
        {
            if (range.empty)
                return;
            range.popFront();
            for (auto entity = range.front; !range.empty; range.popFront(), entity = range.front)
                if (entity.type == EntityType.elementStart && entity.name == "s")
                    break;
            if (!range.empty)
                range.popFront();
        }

        bool empty()
        {
            return range.empty;
        }

    private:
        this(EntityRange!(simpleXML, string) file)
        {
            range = file;
            popFront();
        }

        EntityRange!(simpleXML, string) range = void;
    }

    struct WordTagRange
    {
        Pair front()
        {
            return wordTag;
        }

        void popFront()
        {
            import std.string : stripRight;
            import std.algorithm : find;

            for (auto entity = range.front; !range.empty && !(entity.type == EntityType.elementEnd && entity.name == "s");
                    range.popFront(), entity = range.front)
            {
                if (entity.type == EntityType.elementStart)
                {
                    auto attr = entity.attributes.find!((a, b) => a.name == b)("c5");
                    if (!attr.empty)
                        wordTag.tag = attr.front.value.stripRight();
                }
                else if (entity.type == EntityType.text)
                {
                    wordTag.word = entity.text.stripRight();
                    range.popFront();
                    if (wordTag.tag.length && wordTag.word.length)
                        return;
                }
            }
            empty = true;
        }

        bool empty;

    private:
        this(EntityRange!(simpleXML, string) segment)
        {
            range = segment;
            popFront();
        }

        EntityRange!(simpleXML, string) range = void;
        Pair wordTag;
    }

private:
    EntityRange!(simpleXML, string) parse = void;
    typeof(joiner(bySegment())) pairs = void;
}
}
private static immutable ptbPosGrammar = r"
PtbPos:
    File     <  :(!SentDiv .)* (SentDiv* Sentence SentDiv*)+
	SentDiv  <- :('===' '='+)
    Sentence <  (WordTag / :'[' WordTag+ :']')+
	WordTag  <  ;Word :'/' ;Tag
    Word     <~ Char+
	Tag      <~ Char+
	Char     <~ (:backslash '/')
	          / (!('/' / ' ' / '\t' / '\r' / '\n') .)
";

static immutable string[] ptbPosTags = [
    "#", "$", "``", "''", "(", ")", ",", ".", ":", "CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR",
    "JJS", "LS", "MD", "NN", "NNP", "NNPS", "NNS", "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS",
    "RP", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "WDT", "WP", "WP$", "WRB"
];

version(allparser)
{
struct PtbPosReader
{
    import seqtagsim.reader.grammar.ptb : PtbPos, ParseTree;
    import std.algorithm : countUntil;
    import std.utf : byCodeUnit;

    this(string content)
    {
        auto parse = PtbPos(content);
        if (parse.successful)
        {
            file = parse.children[0];
            pairs = joiner(bySentence);
        }
        else
            throw new Exception("Error parsing file contents!");
    }

@safe @nogc nothrow:

    static immutable fileType = ".pos";
    enum fileFormat = FileFormat.ptbPOS;
    static immutable tagSet = ptbPosTags;

    Pair front()
    {
        return pairs.front;
    }

    void popFront()
    {
        pairs.popFront();
    }

    bool empty()
    {
        return pairs.empty;
    }

    SentenceRange bySentence()
    {
        return SentenceRange(file);
    }

    alias bySegment = bySentence;

    struct WordTagRange
    {
    @safe @nogc nothrow:
        Pair front()
        {
            return pair;
        }

        void popFront()
        {
            wordTags = wordTags[1 .. $];
            buildFront();
        }

        bool empty()
        {
            return pair == Pair.init;
        }

    private:
        ParseTree[] wordTags;
        Pair pair;

        this(ParseTree sentence)
        {
            wordTags = sentence.children;
            buildFront();
        }

        void buildFront()
        {
            if (wordTags.length == 0)
            {
                pair = Pair.init;
                return;
            }
            string word = wordTags[0].matches[0];
            string tags = wordTags[0].matches[1];
            size_t l = tags.byCodeUnit.countUntil('|');
            string tag = tags[0 .. l != -1 ? l : $];
            tag = tag == "JJSS" || tag == "PRP$R" ? tag[0 .. $ - 1] : tag;
            pair = Pair(word, tag);
        }
    }

    struct SentenceRange
    {
    @safe @nogc nothrow:
        WordTagRange front()
        {
            return WordTagRange(sentences[0]);
        }

        void popFront()
        {
            sentences = sentences[1 .. $];
        }

        bool empty()
        {
            return sentences.length == 0;
        }

    private:
        ParseTree[] sentences;

        this(ParseTree file)
        {
            sentences = file.children;
        }
    }

private:
    typeof(joiner(bySentence())) pairs;
    ParseTree file;
}
}

@extends!GenericTsvReader struct GmbPosReader
{
@safe @nogc nothrow:
    mixin base;

    static immutable fileType = ".tags";
    enum fileFormat = FileFormat.gmbPOS;

    this(string content)
    {
        base = GenericTsvReader(content, Config(0, 1, '\t'));
    }
}

@extends!GenericTsvReader struct GmbNerReader
{
@safe @nogc nothrow:
    mixin base;

    static immutable fileType = ".tags";
    enum fileFormat = FileFormat.gmbNER;
    static immutable bTags = ["B-geo", "B-org", "B-per", "B-gpe", "B-tim", "B-art", "B-eve", "B-nat"];
    static immutable iTags = ["I-geo", "I-org", "I-per", "I-gpe", "I-tim", "I-art", "I-eve", "I-nat"];

    this(string content)
    {
        base = GenericTsvReader(content, Config(0, 3, '\t', &convertTag));
    }

    static string convertTag(string tag, string prev) pure
    {
        if (tag.length < 3)
            return tag;
        const string mainTag = tag[0 .. 3];
        if (prev.length >=3 && prev[$-3 .. $] == mainTag)
            switch(mainTag)
            {
                static foreach(t; iTags)
                    case t[2 .. $]: return t;
                default: assert(0, "tag not recognized");
            }
        else
            switch(mainTag)
            {
                static foreach(t; bTags)
                    case t[2 .. $]: return t;
                default: assert(0, "tag not recognized");
            }
    }
}

@extends!GenericTsvReader struct WikiGoldNerReader
{
@safe @nogc nothrow:
    mixin base;

    static immutable fileType = ".txt";
    enum fileFormat = FileFormat.wikigoldNER;
    static immutable bTags = ["B-LOC", "B-ORG", "B-PER", "B-MISC"];

    this(string content)
    {
        base = GenericTsvReader(content, Config(0, 1, ' ', &convertTag));
    }

    static string convertTag(string tag, string prev) pure
    {
        if (tag.length < 5)
            return tag;
        if (prev.length > 3 && tag[2 .. $] == prev[$-3 .. $])
            return tag;
        switch(tag[2 .. $])
        {
            static foreach(b; bTags)
                case b[2 ..  $]: return b;
            default: assert(0, "tag not recognized");
        }
    }
}

@extends!GenericTsvReader struct SecNerReader
{
@safe @nogc nothrow:
    mixin base;

    static immutable fileType = ".secner";
    enum fileFormat = FileFormat.secNER;
    static immutable bTags = ["B-LOC", "B-ORG", "B-PER", "B-MISC"];

    this(string content)
    {
        base = GenericTsvReader(content, Config(0, 3, ' ', &WikiGoldNerReader.convertTag));
    }

}

@extends!GenericTsvReader struct GermanNerReader
{
@safe @nogc nothrow:
    mixin base;

    static immutable fileType = ".tsv";
    enum fileFormat = FileFormat.germanerNER;

    this(string content)
    {
        base = GenericTsvReader(content, Config(1, 2, '\t'));
    }
}

@extends!GenericTsvReader struct WnutConllReader
{
@safe @nogc nothrow:
    mixin base;

    static immutable fileType = ".wnutner";
    enum fileFormat = FileFormat.wnutConll;

    this(string content)
    {
        base = GenericTsvReader(content, Config(0, 1));
    }
}

@extends!GenericTsvReader struct OntonotesConllNerReader
{
@safe @nogc nothrow:
    mixin base;

    static immutable fileType = ".ontoner";
    enum fileFormat = FileFormat.ontoConllNER;

    this(string content)
    {
        base = GenericTsvReader(content, Config(0, 3));
    }
}

@extends!GenericTsvReader struct OntonotesConllPosReader
{
@safe @nogc nothrow:
    mixin base;

    static immutable fileType = ".ontopos";
    enum fileFormat = FileFormat.ontoConllPOS;

    this(string content)
    {
        base = GenericTsvReader(content, Config(0, 1));
    }
}

@extends!GenericTsvReader struct EpConllNerReader
{
@safe @nogc nothrow:
    mixin base;

    static immutable fileType = ".epconll";
    enum fileFormat = FileFormat.epConllNER;
    static immutable bTags = ["B-LOC", "B-ORG", "B-PER", "B-MISC"];
    static immutable iTags = ["I-LOC", "I-ORG", "I-PER", "I-MISC"];

    this(string content)
    {
        base = GenericTsvReader(content, Config(0, 4, ' ', &convertTag));
    }

    static string convertTag(string tag, string prev) pure {
        if (tag.length < 3)
            return tag;
        if (prev.length > 2 && tag == prev[2 .. $]) switch (tag)
        {
            static foreach(i; iTags)
                case i[2 ..  $]: return i;
            default: assert(0, "tag not recognized");
        }
        else switch (tag)
        {
            static foreach(b; bTags)
                case b[2 ..  $]: return b;
            default: assert(0, "tag not recognized");
        }
    }
}

@extends!GenericTsvReader struct ConllNerReader
{
@safe @nogc nothrow:
    mixin base;

    static immutable fileType = ".conll";
    enum fileFormat = FileFormat.conllNER;

    this(string content)
    {
        base = GenericTsvReader(content, Config(0, 3, ' '));
    }
}

@extends!GenericTsvReader struct ConllGerNerReader
{
@safe @nogc nothrow:
    mixin base;

    static immutable fileType = ".conllger";
    enum fileFormat = FileFormat.conllGerNER;
    static immutable bTags = ["B-LOC", "B-ORG", "B-PER", "B-MISC"];
    static immutable iTags = ["I-LOC", "I-ORG", "I-PER", "I-MISC"];

    this(string content)
    {
        base = GenericTsvReader(content[26 ..$], Config(0, 4, ' ', &convertTag));
    }

    static string convertTag(string tag, string prev) pure
    {
        if (tag[0] == 'B') switch (tag[2..$])
        {
            static foreach(i; iTags)
                case i[2 ..  $]: return i;
            default: assert(0, "tag not recognized");
        }
        else if (tag[0] == 'I') switch (tag[2..$])
        {
            static foreach(b; bTags)
                case b[2 ..  $]: return b;
            default: assert(0, "tag not recognized");
        }
        return tag;
    }
}

@extends!GenericTsvReader struct UdPosReader
{
@safe @nogc nothrow:
    mixin base;

    static immutable fileType = ".conllu";
    enum fileFormat = FileFormat.udPOS;
    static immutable tagSet = udPosTags;

    this(string content)
    {
        base = GenericTsvReader(content, Config(1, 3, '\t', &convertTag));
    }

    static string convertTag(string tag, string prev) pure
    {
        return tag == "_" ? udPosTags[$ - 1] : tag;
    }
}

static immutable string[] udPosTags = [
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN",
    "PUNCT", "SCONJ", "SYM", "VERB", "X"
];

struct GenericTsvReader
{
    import std.algorithm : splitter, skipOver, filter;
    import std.range : dropExactly;
    import std.utf : byCodeUnit;
    import std.uni : isWhite;
    import std.string : lineSplitter;

@safe @nogc nothrow:

    this(string content, Config config)
    {
        this.config = config;
        this.text = content;
        this.pairs = joiner(bySentence());
    }

    Pair front()
    {
        return pairs.front;
    }

    void popFront()
    {
        pairs.popFront();
    }

    bool empty()
    {
        return pairs.empty;
    }

    SentenceRange bySentence()
    {
        return SentenceRange(text, config);
    }

    alias bySegment = bySentence;

    struct SentenceRange
    {
    @safe @nogc nothrow:

        WordTagRange front()
        {
            return WordTagRange(lines, config);
        }

        void popFront()
        {
            lines.skipOver!(a => !a.filter!(a => !a.isWhite).empty);
            lines.skipOver!(a => a.filter!(a => !a.isWhite).empty || a.front == '#');
        }

        bool empty()
        {
            return lines.empty();
        }

    private:
        LineSplitter lines;
        Config config;

        this(string text, ref Config config)
        {
            this.config = config;
            lines = text.byCodeUnit.lineSplitter;
            lines.skipOver!(a => a.front == '#');
        }
    }

    struct WordTagRange
    {
    @safe @nogc nothrow:

        Pair front()
        {
            return pair;
        }

        void popFront()
        {
            if (lines.empty || lines.front.empty || lines.front.filter!(a => !a.isWhite).empty)
            {
                pair = Pair.init;
                return;
            }
            auto tokens = lines.front.splitter(fieldSep).filter!(a => !a.empty).dropExactly(wordColumn);
            pair.word = tokens.front.source;
            string tagToBe = tokens.dropExactly(labelColumn - wordColumn).front.source;
            pair.tag = tagMapping(tagToBe, lastTag);
            lastTag = pair.tag;
            lines.popFront();
        }

        bool empty()
        {
            return pair == Pair.init;
        }

    private:
        LineSplitter lines;
        Pair pair;
        string lastTag;
        Config config;

        alias config this;

        this(ref LineSplitter lines, ref Config config)
        {
            this.config = config;
            this.lines = lines;
            popFront();
        }
    }

private:
    alias LineSplitter = typeof("".byCodeUnit.lineSplitter);
    typeof(joiner(bySentence())) pairs;
    string text;
    Config config;

    struct Config
    {
        int wordColumn;
        int labelColumn;
        char fieldSep = '\t';
        string function(string, string) @nogc nothrow pure tagMapping = (a,b) => a;
    }
}
