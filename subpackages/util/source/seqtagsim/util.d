/**
 * Copyright: Fynn Schröder, 2019
 * Author: Fynn Schröder
 * License: MIT
 */

module seqtagsim.util;

/**
 * Lazily iterates files matching the pattern.
 *
 * Params:
 *     start = path of directory to use as starting point for matching
 *     pattern = filter pattern
 *
 * Returns:
 *     InputRange of files matching the pattern
 */
auto listFiles(string start, string pattern = null)
{
    import std.file : dirEntries, SpanMode, isDir;
    import std.range : only, inputRangeObject, InputRange;
    import std.algorithm.iteration : filter, map;

    if (isDir(start))
    {
        if (pattern is null)
            return cast(InputRange!string) dirEntries(start, SpanMode.depth).filter!(e => e.isFile)
                .map!(f => f.name)
                .inputRangeObject;
        return cast(InputRange!string) dirEntries(start, pattern, SpanMode.depth).filter!(e => e.isFile)
            .map!(f => f.name)
            .inputRangeObject;
    }
    return cast(InputRange!string) only(start).inputRangeObject;
}

/**
 * Generic output buffer / output range to add one item after the other.
 */
struct OutputBuffer(T, Allocator)
{
    import std.experimental.allocator : makeArray, expandArray, dispose, stateSize, shrinkArray;
    import std.algorithm.mutation : copy;

    @disable this(this);

    alias data this;

    static if (stateSize!Allocator != 0)
    {
        @disable this();
        this(ref Allocator allocator, size_t initialCapacity = defaultCapacity)
        {
            alloc = &allocator;
            buf = makeArray!T(alloc, initialCapacity);
        }
    }
    else
    {
        this(size_t initialCapacity)
        {
            buf = makeArray!T(alloc, initialCapacity);
        }
    }

    void put(const T elem)
    {
        if (buf.length == 0)
            buf = makeArray!T(alloc, defaultCapacity);
        else if (buf.length == len)
            expandArray(alloc, buf, buf.length);
        buf[len++] = elem;
    }

    void opOpAssign(string op)(const T elem) if (op == "~")
    {
        put(elem);
    }

    void opIndexAssign(const T elem, size_t index)
    {
        buf[index] = elem;
    }

    T[] data()
    {
        return buf[0 .. len];
    }

    void minimize()
    {
        shrinkArray(alloc, buf, buf.length - len);
    }

    void remove(size_t from, size_t to)
    {
        immutable newLen = len - (to - from);
        copy(buf[to .. len], buf[from .. newLen]);
        len = newLen;
    }

    size_t length()
    {
        return len;
    }

    ~this()
    {
        dispose(alloc, buf);
    }

private:
    enum defaultCapacity = 16;
    static if (stateSize!Allocator == 0)
        alias alloc = Allocator.instance;
    else
        Allocator* alloc;
    size_t len;
    T[] buf;
}

unittest
{
    import std.experimental.allocator.mallocator : Mallocator;
    import std.range : put;

    auto o = OutputBuffer!(char, Mallocator)();
    string dummyData = "00112233445566778899";
    put(o, dummyData);
    assert(dummyData == o.data);
    o.remove(8, 10);
    assert(o.data == "001122335566778899");
}

unittest
{
    import std.experimental.allocator.building_blocks.region : Region;
    import std.experimental.allocator.mallocator : Mallocator;
    import std.range : put;

    Region!Mallocator alloc = Region!Mallocator(64);

    auto o = OutputBuffer!(char, typeof(alloc))(alloc);
    string dummyData = "00112233445566778899";
    put(o, dummyData);
    assert(dummyData == o.data);
}

/**
 * Pretty prints a struct recursively using its field names.
 *
 * Params:
 *     s = structure
 */
void prettyPrintStruct(S)(S s)
{
    import std.traits : FieldNameTuple;
    import std.stdio : write, writefln;
    import std.uni : isUpper, toUpper;

    foreach (index, name; FieldNameTuple!S)
    {
        write(toUpper(name[0]));
        foreach (char c; name[1 .. $])
        {
            if (isUpper(c))
                write(' ');
            write(c);
        }
        writefln(": %s", s.tupleof[index]);
    }
}

/// UDA to mark UDAs
@UDA struct UDA
{
}

/// UDA showing static inheritance of this struct to T (use with `mixin base;`)
@UDA struct extends(T) if (is(T == struct))
{
}

/// Mixin this template for static inheritance of the base type specified by UDA `extends(T)`
mixin template base()
{
    import std.traits : getUDAs, TemplateArgsOf;

    TemplateArgsOf!(getUDAs!(typeof(this), extends)[0])[0] base;
    alias base this;
}

/// Atomic wrapper for the integral types that support atomic opeartions.
struct Atomic(T)
{
    import core.atomic : atomicOp, atomicStore, atomicLoad;

    this(const T value)
    {
        opAssign(value);
    }

    auto opOpAssign(string op)(const T value)
    {
        return atomicOp!(op ~ '=')(number, value);
    }

    void opAssign(const T value)
    {
        atomicStore(number, value);
    }

    T get() const
    {
        return atomicLoad(number);
    }

    alias get this;

    private shared T number;
}

unittest
{
    Atomic!uint a = 1;
    a++;
    a += 10;
    a -= 3;
    assert(a == 9);
}

/// Progress monitor and printer.
struct Progress
{
    import std.datetime.stopwatch : StopWatch, AutoStart;
    import std.stdio : stderr;
    import core.atomic : atomicOp;

    StopWatch sw;
    alias sw this;

    @disable this();

    this(const ulong total)
    {
        this.total = total;
        printEvery = total / 1000;
        current = printEvery - 1;
        sw = StopWatch(AutoStart.yes);
    }

    void opUnary(string op)() if (op == "++")
    {
        advance(1);
    }

    void opOpAssign(string op)(const ulong amount) if (op == "+")
    {
        advance(amount);
    }

    bool isComplete()
    {
        return progress == total;
    }

    void reset()
    {
        current = printEvery - 1;
        progress = 0;
    }

private:
    static immutable string message = "\rComparison progress % 5.1f %%";
    ulong total;
    ulong printEvery;
    Atomic!ulong current;
    Atomic!ulong progress;
    ulong nextTime;

    void advance(const ulong amount)
    {
        if ((progress += amount) == total)
        {
            sw.stop();
            stderr.writefln!message(100.0);
        }
        else if ((current += amount) >= printEvery)
        {
            current = 0;
            immutable ulong now = sw.peek.total!"msecs";
            if (now > nextTime)
            {
                nextTime = now + 1000;
                stderr.writef!message(100.0 * progress / total);
                stderr.flush();
            }
        }
    }
}

unittest
{
    Progress p = Progress(4);
    p++;
    p += 2;
    assert(!p.isComplete);
    p++;
    assert(p.isComplete);
    p.reset();
    assert(!p.isComplete);
    p += 4;
    assert(p.isComplete);
}
