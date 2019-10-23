/++
This module was automatically generated from the following grammar:


PtbPos:
    File     <  :(!SentDiv .)* (SentDiv* Sentence SentDiv*)+
	SentDiv  <- :('===' '='+)
    Sentence <  (WordTag / :'[' WordTag+ :']')+
	WordTag  <  ;Word :'/' ;Tag
    Word     <~ Char+
	Tag      <~ Char+
	Char     <~ (:backslash '/')
	          / (!('/' / ' ' / '\t' / '\r' / '\n') .)


+/
module seqtagsim.reader.grammar.ptb;

version(allparser)
{

public import pegged.peg;
import std.algorithm: startsWith;
import std.functional: toDelegate;

struct GenericPtbPos(TParseTree)
{
    import std.functional : toDelegate;
    import pegged.dynamic.grammar;
    static import pegged.peg;
    struct PtbPos
    {
    enum name = "PtbPos";
    static ParseTree delegate(ParseTree)[string] before;
    static ParseTree delegate(ParseTree)[string] after;
    static ParseTree delegate(ParseTree)[string] rules;
    import std.typecons:Tuple, tuple;
    static TParseTree[Tuple!(string, size_t)] memo;
    static this()
    {
        rules["File"] = toDelegate(&File);
        rules["SentDiv"] = toDelegate(&SentDiv);
        rules["Sentence"] = toDelegate(&Sentence);
        rules["WordTag"] = toDelegate(&WordTag);
        rules["Word"] = toDelegate(&Word);
        rules["Tag"] = toDelegate(&Tag);
        rules["Char"] = toDelegate(&Char);
        rules["Spacing"] = toDelegate(&Spacing);
    }

    template hooked(alias r, string name)
    {
        static ParseTree hooked(ParseTree p)
        {
            ParseTree result;

            if (name in before)
            {
                result = before[name](p);
                if (result.successful)
                    return result;
            }

            result = r(p);
            if (result.successful || name !in after)
                return result;

            result = after[name](p);
            return result;
        }

        static ParseTree hooked(string input)
        {
            return hooked!(r, name)(ParseTree("",false,[],input));
        }
    }

    static void addRuleBefore(string parentRule, string ruleSyntax)
    {
        // enum name is the current grammar name
        DynamicGrammar dg = pegged.dynamic.grammar.grammar(name ~ ": " ~ ruleSyntax, rules);
        foreach(ruleName,rule; dg.rules)
            if (ruleName != "Spacing") // Keep the local Spacing rule, do not overwrite it
                rules[ruleName] = rule;
        before[parentRule] = rules[dg.startingRule];
    }

    static void addRuleAfter(string parentRule, string ruleSyntax)
    {
        // enum name is the current grammar named
        DynamicGrammar dg = pegged.dynamic.grammar.grammar(name ~ ": " ~ ruleSyntax, rules);
        foreach(name,rule; dg.rules)
        {
            if (name != "Spacing")
                rules[name] = rule;
        }
        after[parentRule] = rules[dg.startingRule];
    }

    static bool isRule(string s)
    {
		import std.algorithm : startsWith;
        return s.startsWith("PtbPos.");
    }
    mixin decimateTree;

    alias spacing Spacing;

    static TParseTree File(TParseTree p)
    {
        if(__ctfe)
        {
            return         pegged.peg.defined!(pegged.peg.and!(pegged.peg.discard!(pegged.peg.zeroOrMore!(pegged.peg.wrapAround!(Spacing, pegged.peg.and!(pegged.peg.negLookahead!(pegged.peg.wrapAround!(Spacing, SentDiv, Spacing)), pegged.peg.wrapAround!(Spacing, pegged.peg.any, Spacing)), Spacing))), pegged.peg.oneOrMore!(pegged.peg.wrapAround!(Spacing, pegged.peg.and!(pegged.peg.zeroOrMore!(pegged.peg.wrapAround!(Spacing, SentDiv, Spacing)), pegged.peg.wrapAround!(Spacing, Sentence, Spacing), pegged.peg.zeroOrMore!(pegged.peg.wrapAround!(Spacing, SentDiv, Spacing))), Spacing))), "PtbPos.File")(p);
        }
        else
        {
            if (auto m = tuple(`File`, p.end) in memo)
                return *m;
            else
            {
                TParseTree result = hooked!(pegged.peg.defined!(pegged.peg.and!(pegged.peg.discard!(pegged.peg.zeroOrMore!(pegged.peg.wrapAround!(Spacing, pegged.peg.and!(pegged.peg.negLookahead!(pegged.peg.wrapAround!(Spacing, SentDiv, Spacing)), pegged.peg.wrapAround!(Spacing, pegged.peg.any, Spacing)), Spacing))), pegged.peg.oneOrMore!(pegged.peg.wrapAround!(Spacing, pegged.peg.and!(pegged.peg.zeroOrMore!(pegged.peg.wrapAround!(Spacing, SentDiv, Spacing)), pegged.peg.wrapAround!(Spacing, Sentence, Spacing), pegged.peg.zeroOrMore!(pegged.peg.wrapAround!(Spacing, SentDiv, Spacing))), Spacing))), "PtbPos.File"), "File")(p);
                memo[tuple(`File`, p.end)] = result;
                return result;
            }
        }
    }

    static TParseTree File(string s)
    {
        if(__ctfe)
        {
            return         pegged.peg.defined!(pegged.peg.and!(pegged.peg.discard!(pegged.peg.zeroOrMore!(pegged.peg.wrapAround!(Spacing, pegged.peg.and!(pegged.peg.negLookahead!(pegged.peg.wrapAround!(Spacing, SentDiv, Spacing)), pegged.peg.wrapAround!(Spacing, pegged.peg.any, Spacing)), Spacing))), pegged.peg.oneOrMore!(pegged.peg.wrapAround!(Spacing, pegged.peg.and!(pegged.peg.zeroOrMore!(pegged.peg.wrapAround!(Spacing, SentDiv, Spacing)), pegged.peg.wrapAround!(Spacing, Sentence, Spacing), pegged.peg.zeroOrMore!(pegged.peg.wrapAround!(Spacing, SentDiv, Spacing))), Spacing))), "PtbPos.File")(TParseTree("", false,[], s));
        }
        else
        {
            forgetMemo();
            return hooked!(pegged.peg.defined!(pegged.peg.and!(pegged.peg.discard!(pegged.peg.zeroOrMore!(pegged.peg.wrapAround!(Spacing, pegged.peg.and!(pegged.peg.negLookahead!(pegged.peg.wrapAround!(Spacing, SentDiv, Spacing)), pegged.peg.wrapAround!(Spacing, pegged.peg.any, Spacing)), Spacing))), pegged.peg.oneOrMore!(pegged.peg.wrapAround!(Spacing, pegged.peg.and!(pegged.peg.zeroOrMore!(pegged.peg.wrapAround!(Spacing, SentDiv, Spacing)), pegged.peg.wrapAround!(Spacing, Sentence, Spacing), pegged.peg.zeroOrMore!(pegged.peg.wrapAround!(Spacing, SentDiv, Spacing))), Spacing))), "PtbPos.File"), "File")(TParseTree("", false,[], s));
        }
    }
    static string File(GetName g)
    {
        return "PtbPos.File";
    }

    static TParseTree SentDiv(TParseTree p)
    {
        if(__ctfe)
        {
            return         pegged.peg.defined!(pegged.peg.discard!(pegged.peg.and!(pegged.peg.literal!("==="), pegged.peg.oneOrMore!(pegged.peg.literal!("=")))), "PtbPos.SentDiv")(p);
        }
        else
        {
            if (auto m = tuple(`SentDiv`, p.end) in memo)
                return *m;
            else
            {
                TParseTree result = hooked!(pegged.peg.defined!(pegged.peg.discard!(pegged.peg.and!(pegged.peg.literal!("==="), pegged.peg.oneOrMore!(pegged.peg.literal!("=")))), "PtbPos.SentDiv"), "SentDiv")(p);
                memo[tuple(`SentDiv`, p.end)] = result;
                return result;
            }
        }
    }

    static TParseTree SentDiv(string s)
    {
        if(__ctfe)
        {
            return         pegged.peg.defined!(pegged.peg.discard!(pegged.peg.and!(pegged.peg.literal!("==="), pegged.peg.oneOrMore!(pegged.peg.literal!("=")))), "PtbPos.SentDiv")(TParseTree("", false,[], s));
        }
        else
        {
            forgetMemo();
            return hooked!(pegged.peg.defined!(pegged.peg.discard!(pegged.peg.and!(pegged.peg.literal!("==="), pegged.peg.oneOrMore!(pegged.peg.literal!("=")))), "PtbPos.SentDiv"), "SentDiv")(TParseTree("", false,[], s));
        }
    }
    static string SentDiv(GetName g)
    {
        return "PtbPos.SentDiv";
    }

    static TParseTree Sentence(TParseTree p)
    {
        if(__ctfe)
        {
            return         pegged.peg.defined!(pegged.peg.oneOrMore!(pegged.peg.wrapAround!(Spacing, pegged.peg.or!(pegged.peg.wrapAround!(Spacing, WordTag, Spacing), pegged.peg.and!(pegged.peg.discard!(pegged.peg.wrapAround!(Spacing, pegged.peg.literal!("["), Spacing)), pegged.peg.oneOrMore!(pegged.peg.wrapAround!(Spacing, WordTag, Spacing)), pegged.peg.discard!(pegged.peg.wrapAround!(Spacing, pegged.peg.literal!("]"), Spacing)))), Spacing)), "PtbPos.Sentence")(p);
        }
        else
        {
            if (auto m = tuple(`Sentence`, p.end) in memo)
                return *m;
            else
            {
                TParseTree result = hooked!(pegged.peg.defined!(pegged.peg.oneOrMore!(pegged.peg.wrapAround!(Spacing, pegged.peg.or!(pegged.peg.wrapAround!(Spacing, WordTag, Spacing), pegged.peg.and!(pegged.peg.discard!(pegged.peg.wrapAround!(Spacing, pegged.peg.literal!("["), Spacing)), pegged.peg.oneOrMore!(pegged.peg.wrapAround!(Spacing, WordTag, Spacing)), pegged.peg.discard!(pegged.peg.wrapAround!(Spacing, pegged.peg.literal!("]"), Spacing)))), Spacing)), "PtbPos.Sentence"), "Sentence")(p);
                memo[tuple(`Sentence`, p.end)] = result;
                return result;
            }
        }
    }

    static TParseTree Sentence(string s)
    {
        if(__ctfe)
        {
            return         pegged.peg.defined!(pegged.peg.oneOrMore!(pegged.peg.wrapAround!(Spacing, pegged.peg.or!(pegged.peg.wrapAround!(Spacing, WordTag, Spacing), pegged.peg.and!(pegged.peg.discard!(pegged.peg.wrapAround!(Spacing, pegged.peg.literal!("["), Spacing)), pegged.peg.oneOrMore!(pegged.peg.wrapAround!(Spacing, WordTag, Spacing)), pegged.peg.discard!(pegged.peg.wrapAround!(Spacing, pegged.peg.literal!("]"), Spacing)))), Spacing)), "PtbPos.Sentence")(TParseTree("", false,[], s));
        }
        else
        {
            forgetMemo();
            return hooked!(pegged.peg.defined!(pegged.peg.oneOrMore!(pegged.peg.wrapAround!(Spacing, pegged.peg.or!(pegged.peg.wrapAround!(Spacing, WordTag, Spacing), pegged.peg.and!(pegged.peg.discard!(pegged.peg.wrapAround!(Spacing, pegged.peg.literal!("["), Spacing)), pegged.peg.oneOrMore!(pegged.peg.wrapAround!(Spacing, WordTag, Spacing)), pegged.peg.discard!(pegged.peg.wrapAround!(Spacing, pegged.peg.literal!("]"), Spacing)))), Spacing)), "PtbPos.Sentence"), "Sentence")(TParseTree("", false,[], s));
        }
    }
    static string Sentence(GetName g)
    {
        return "PtbPos.Sentence";
    }

    static TParseTree WordTag(TParseTree p)
    {
        if(__ctfe)
        {
            return         pegged.peg.defined!(pegged.peg.and!(pegged.peg.drop!(pegged.peg.wrapAround!(Spacing, Word, Spacing)), pegged.peg.discard!(pegged.peg.wrapAround!(Spacing, pegged.peg.literal!("/"), Spacing)), pegged.peg.drop!(pegged.peg.wrapAround!(Spacing, Tag, Spacing))), "PtbPos.WordTag")(p);
        }
        else
        {
            if (auto m = tuple(`WordTag`, p.end) in memo)
                return *m;
            else
            {
                TParseTree result = hooked!(pegged.peg.defined!(pegged.peg.and!(pegged.peg.drop!(pegged.peg.wrapAround!(Spacing, Word, Spacing)), pegged.peg.discard!(pegged.peg.wrapAround!(Spacing, pegged.peg.literal!("/"), Spacing)), pegged.peg.drop!(pegged.peg.wrapAround!(Spacing, Tag, Spacing))), "PtbPos.WordTag"), "WordTag")(p);
                memo[tuple(`WordTag`, p.end)] = result;
                return result;
            }
        }
    }

    static TParseTree WordTag(string s)
    {
        if(__ctfe)
        {
            return         pegged.peg.defined!(pegged.peg.and!(pegged.peg.drop!(pegged.peg.wrapAround!(Spacing, Word, Spacing)), pegged.peg.discard!(pegged.peg.wrapAround!(Spacing, pegged.peg.literal!("/"), Spacing)), pegged.peg.drop!(pegged.peg.wrapAround!(Spacing, Tag, Spacing))), "PtbPos.WordTag")(TParseTree("", false,[], s));
        }
        else
        {
            forgetMemo();
            return hooked!(pegged.peg.defined!(pegged.peg.and!(pegged.peg.drop!(pegged.peg.wrapAround!(Spacing, Word, Spacing)), pegged.peg.discard!(pegged.peg.wrapAround!(Spacing, pegged.peg.literal!("/"), Spacing)), pegged.peg.drop!(pegged.peg.wrapAround!(Spacing, Tag, Spacing))), "PtbPos.WordTag"), "WordTag")(TParseTree("", false,[], s));
        }
    }
    static string WordTag(GetName g)
    {
        return "PtbPos.WordTag";
    }

    static TParseTree Word(TParseTree p)
    {
        if(__ctfe)
        {
            return         pegged.peg.defined!(pegged.peg.fuse!(pegged.peg.oneOrMore!(Char)), "PtbPos.Word")(p);
        }
        else
        {
            if (auto m = tuple(`Word`, p.end) in memo)
                return *m;
            else
            {
                TParseTree result = hooked!(pegged.peg.defined!(pegged.peg.fuse!(pegged.peg.oneOrMore!(Char)), "PtbPos.Word"), "Word")(p);
                memo[tuple(`Word`, p.end)] = result;
                return result;
            }
        }
    }

    static TParseTree Word(string s)
    {
        if(__ctfe)
        {
            return         pegged.peg.defined!(pegged.peg.fuse!(pegged.peg.oneOrMore!(Char)), "PtbPos.Word")(TParseTree("", false,[], s));
        }
        else
        {
            forgetMemo();
            return hooked!(pegged.peg.defined!(pegged.peg.fuse!(pegged.peg.oneOrMore!(Char)), "PtbPos.Word"), "Word")(TParseTree("", false,[], s));
        }
    }
    static string Word(GetName g)
    {
        return "PtbPos.Word";
    }

    static TParseTree Tag(TParseTree p)
    {
        if(__ctfe)
        {
            return         pegged.peg.defined!(pegged.peg.fuse!(pegged.peg.oneOrMore!(Char)), "PtbPos.Tag")(p);
        }
        else
        {
            if (auto m = tuple(`Tag`, p.end) in memo)
                return *m;
            else
            {
                TParseTree result = hooked!(pegged.peg.defined!(pegged.peg.fuse!(pegged.peg.oneOrMore!(Char)), "PtbPos.Tag"), "Tag")(p);
                memo[tuple(`Tag`, p.end)] = result;
                return result;
            }
        }
    }

    static TParseTree Tag(string s)
    {
        if(__ctfe)
        {
            return         pegged.peg.defined!(pegged.peg.fuse!(pegged.peg.oneOrMore!(Char)), "PtbPos.Tag")(TParseTree("", false,[], s));
        }
        else
        {
            forgetMemo();
            return hooked!(pegged.peg.defined!(pegged.peg.fuse!(pegged.peg.oneOrMore!(Char)), "PtbPos.Tag"), "Tag")(TParseTree("", false,[], s));
        }
    }
    static string Tag(GetName g)
    {
        return "PtbPos.Tag";
    }

    static TParseTree Char(TParseTree p)
    {
        if(__ctfe)
        {
            return         pegged.peg.defined!(pegged.peg.fuse!(pegged.peg.or!(pegged.peg.and!(pegged.peg.discard!(backslash), pegged.peg.literal!("/")), pegged.peg.and!(pegged.peg.negLookahead!(pegged.peg.keywords!("/", " ", "\t", "\r", "\n")), pegged.peg.any))), "PtbPos.Char")(p);
        }
        else
        {
            if (auto m = tuple(`Char`, p.end) in memo)
                return *m;
            else
            {
                TParseTree result = hooked!(pegged.peg.defined!(pegged.peg.fuse!(pegged.peg.or!(pegged.peg.and!(pegged.peg.discard!(backslash), pegged.peg.literal!("/")), pegged.peg.and!(pegged.peg.negLookahead!(pegged.peg.keywords!("/", " ", "\t", "\r", "\n")), pegged.peg.any))), "PtbPos.Char"), "Char")(p);
                memo[tuple(`Char`, p.end)] = result;
                return result;
            }
        }
    }

    static TParseTree Char(string s)
    {
        if(__ctfe)
        {
            return         pegged.peg.defined!(pegged.peg.fuse!(pegged.peg.or!(pegged.peg.and!(pegged.peg.discard!(backslash), pegged.peg.literal!("/")), pegged.peg.and!(pegged.peg.negLookahead!(pegged.peg.keywords!("/", " ", "\t", "\r", "\n")), pegged.peg.any))), "PtbPos.Char")(TParseTree("", false,[], s));
        }
        else
        {
            forgetMemo();
            return hooked!(pegged.peg.defined!(pegged.peg.fuse!(pegged.peg.or!(pegged.peg.and!(pegged.peg.discard!(backslash), pegged.peg.literal!("/")), pegged.peg.and!(pegged.peg.negLookahead!(pegged.peg.keywords!("/", " ", "\t", "\r", "\n")), pegged.peg.any))), "PtbPos.Char"), "Char")(TParseTree("", false,[], s));
        }
    }
    static string Char(GetName g)
    {
        return "PtbPos.Char";
    }

    static TParseTree opCall(TParseTree p)
    {
        TParseTree result = decimateTree(File(p));
        result.children = [result];
        result.name = "PtbPos";
        return result;
    }

    static TParseTree opCall(string input)
    {
        if(__ctfe)
        {
            return PtbPos(TParseTree(``, false, [], input, 0, 0));
        }
        else
        {
            forgetMemo();
            return PtbPos(TParseTree(``, false, [], input, 0, 0));
        }
    }
    static string opCall(GetName g)
    {
        return "PtbPos";
    }


    static void forgetMemo()
    {
        memo = null;
    }
    }
}

alias GenericPtbPos!(ParseTree).PtbPos PtbPos;

}