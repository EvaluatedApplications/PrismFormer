namespace PrismGym;

/// <summary>An example (a prompt the model reads → the completion it should learn to write) and a held-out probe
/// (a prompt → the value-equivalent answers that count as correct). Prompts/completions are plain text; the gym
/// tokenises them and trains next-token over the whole sequence.</summary>
public readonly record struct Example(string Prompt, string Completion);
public readonly record struct Probe(string Prompt, string[] Answers);

/// <summary>A trainable skill: it makes training examples and held-out probes at a difficulty level. Skills are
/// self-contained generators — no model, no substrate — they just produce text pairs.</summary>
public interface ISkill
{
    string Name { get; }
    Example Train(Random rng, int level);
    Probe Probe(Random rng, int level);
}

/// <summary>Whole-number arithmetic. The operand range grows with level; the answer is a single number token, so a
/// correct model must actually compute it (held-out operands prove it isn't memorising).</summary>
public sealed class Arithmetic : ISkill
{
    private readonly char _op;
    public Arithmetic(char op) => _op = op;
    public string Name => _op switch { '+' => "add", '-' => "sub", '*' => "mul", _ => "div" };

    private (int a, int b, long c) Draw(Random rng, int level)
    {
        var hi = 5 + level * 5;                      // range grows with level
        int a = rng.Next(0, hi), b = rng.Next(1, hi);
        return _op switch
        {
            '+' => (a, b, (long)a + b),
            '-' => (a + b, b, a),                     // keep the result non-negative
            '*' => (a, b, (long)a * b),
            _ => (a * b, b, a),                       // exact division
        };
    }
    private static Example Frame(int a, char op, int b, long c) => new($"{a} {op} {b} =", c.ToString());
    public Example Train(Random rng, int level) { var (a, b, c) = Draw(rng, level); return Frame(a, _op, b, c); }
    public Probe Probe(Random rng, int level) { var (a, b, c) = Draw(rng, level); return new($"{a} {_op} {b} =", new[] { c.ToString() }); }
}

/// <summary>Digit ↔ word (0..99). Both directions, so the model learns the lexicon.</summary>
public sealed class NumberWords : ISkill
{
    private static readonly string[] Ones = { "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
        "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen" };
    private static readonly string[] Tens = { "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety" };
    public string Name => "numwords";
    private static string Word(int n) => n < 20 ? Ones[n] : Tens[n / 10] + (n % 10 == 0 ? "" : " " + Ones[n % 10]);

    private (int n, bool toWord) Draw(Random rng, int level) => (rng.Next(0, Math.Min(100, 20 + level * 20)), rng.Next(2) == 0);
    private Example Frame(int n, bool toWord) => toWord ? new($"{n} in words is", Word(n)) : new($"the number {Word(n)} is", n.ToString());
    public Example Train(Random rng, int level) { var (n, w) = Draw(rng, level); return Frame(n, w); }
    public Probe Probe(Random rng, int level) { var (n, w) = Draw(rng, level); var e = Frame(n, w); return new(e.Prompt, new[] { e.Completion }); }
}

/// <summary>Copy the token at a fixed position — a pure routing/attention skill (no arithmetic).</summary>
public sealed class Copy : ISkill
{
    private static readonly string[] Vocab = { "red", "blue", "dog", "cat", "sun", "moon", "tree", "fish", "gold", "rock", "star", "leaf" };
    public string Name => "copy";
    private (string[] toks, int pick) Draw(Random rng, int level)
    {
        var n = 3 + Math.Min(level, 4);
        var toks = Enumerable.Range(0, n).Select(_ => Vocab[rng.Next(Vocab.Length)]).ToArray();
        return (toks, 0);                            // recall the FIRST token
    }
    private static Example Frame(string[] toks, int pick) => new($"repeat the first of {string.Join(' ', toks)} :", toks[pick]);
    public Example Train(Random rng, int level) { var (t, p) = Draw(rng, level); return Frame(t, p); }
    public Probe Probe(Random rng, int level) { var (t, p) = Draw(rng, level); return new($"repeat the first of {string.Join(' ', t)} :", new[] { t[p] }); }
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
// Skills ported from the GenesisNova app's data creators / gym language facts (self-contained: the word lists,
// categories, relation facts and sequences are embedded directly — no dependency on GenesisNova). Each frames a
// short prompt → single-token answer so the gym can grade on the first generated token.
// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────

/// <summary>Category membership (is-a): item → its category ("apple" → "fruit"). A retrieval/classification skill —
/// the answer is a learned fact, not computed. Source: GenesisNova CategoryRetrievalCreator / GymLanguageFacts.Categories.</summary>
public sealed class Category : ISkill
{
    private static readonly (string Item, string Cat)[] Table =
    {
        ("apple","fruit"),("banana","fruit"),("grape","fruit"),("pear","fruit"),("cherry","fruit"),("peach","fruit"),("plum","fruit"),("lemon","fruit"),("mango","fruit"),
        ("carrot","vegetable"),("potato","vegetable"),("onion","vegetable"),("pea","vegetable"),("corn","vegetable"),("celery","vegetable"),
        ("dog","animal"),("cat","animal"),("horse","animal"),("cow","animal"),("pig","animal"),("sheep","animal"),("goat","animal"),("lion","animal"),("tiger","animal"),("bear","animal"),("wolf","animal"),("fox","animal"),
        ("robin","bird"),("eagle","bird"),("sparrow","bird"),("owl","bird"),("hawk","bird"),("crow","bird"),("duck","bird"),("goose","bird"),
        ("red","color"),("blue","color"),("green","color"),("yellow","color"),("purple","color"),("pink","color"),("brown","color"),("black","color"),
        ("iron","metal"),("gold","metal"),("silver","metal"),("copper","metal"),("tin","metal"),("zinc","metal"),
        ("car","vehicle"),("truck","vehicle"),("bus","vehicle"),("train","vehicle"),("plane","vehicle"),("boat","vehicle"),("ship","vehicle"),("van","vehicle"),
        ("chair","furniture"),("table","furniture"),("bed","furniture"),("desk","furniture"),("sofa","furniture"),("shelf","furniture"),("stool","furniture"),
        ("piano","instrument"),("guitar","instrument"),("drum","instrument"),("violin","instrument"),("flute","instrument"),("trumpet","instrument"),
        ("shirt","clothing"),("hat","clothing"),("coat","clothing"),("sock","clothing"),("glove","clothing"),("scarf","clothing"),
    };
    private static readonly string[] Frames = { "{0} is a", "classify {0}", "{0} belongs to", "the category of {0} is" };
    public string Name => "category";
    private (string item, string cat) Draw(Random rng, int level) => Table[rng.Next(Math.Min(Table.Length, 16 + level * 8))];
    public Example Train(Random rng, int level) { var (i, c) = Draw(rng, level); return new(string.Format(Frames[rng.Next(Frames.Length)], i), c); }
    public Probe Probe(Random rng, int level) { var (i, c) = Draw(rng, level); return new($"{i} is a", new[] { c }); }
}

/// <summary>Synonyms: a cue word → a word that means the same ("big" → large/huge/giant). Many correct answers (any
/// other member of the group counts). Source: GenesisNova GymLanguageFacts.SynonymGroups.</summary>
public sealed class Synonym : ISkill
{
    private static readonly string[][] Groups =
    {
        new[]{"big","large","huge","giant","enormous"}, new[]{"small","little","tiny","miniature","petite"},
        new[]{"happy","glad","cheerful","joyful","pleased"}, new[]{"sad","unhappy","gloomy","miserable","downcast"},
        new[]{"fast","quick","rapid","swift","speedy"}, new[]{"slow","sluggish","leisurely","unhurried"},
        new[]{"smart","clever","bright","intelligent","sharp"}, new[]{"angry","mad","furious","irate","livid"},
        new[]{"begin","start","commence","initiate"}, new[]{"end","finish","conclude","complete"},
        new[]{"cold","chilly","freezing","frigid","icy"}, new[]{"quiet","silent","hushed","soundless"},
        new[]{"loud","noisy","deafening","blaring"}, new[]{"easy","simple","effortless","painless"},
        new[]{"hard","difficult","tough","challenging"}, new[]{"strong","powerful","mighty","sturdy"},
        new[]{"rich","wealthy","affluent","prosperous"}, new[]{"old","ancient","aged","elderly"},
        new[]{"new","modern","fresh","recent"}, new[]{"pretty","beautiful","lovely","gorgeous"},
        new[]{"brave","bold","fearless","courageous"}, new[]{"funny","amusing","hilarious","comical"},
        new[]{"clean","spotless","immaculate","pristine"}, new[]{"dirty","filthy","grimy","muddy"},
        new[]{"tired","weary","exhausted","drained"}, new[]{"calm","peaceful","serene","tranquil"},
    };
    private static readonly string[] Frames = { "a synonym for {0}", "another word for {0}", "what means the same as {0}" };
    public string Name => "synonym";
    private string[] Draw(Random rng, int level) => Groups[rng.Next(Math.Min(Groups.Length, 10 + level * 4))];
    public Example Train(Random rng, int level)
    {
        var g = Draw(rng, level); var cue = rng.Next(g.Length);
        int ans; do { ans = rng.Next(g.Length); } while (ans == cue);
        return new(string.Format(Frames[rng.Next(Frames.Length)], g[cue]), g[ans]);
    }
    public Probe Probe(Random rng, int level)
    {
        var g = Draw(rng, level); var cue = rng.Next(g.Length);
        return new($"a synonym for {g[cue]}", g.Where((_, i) => i != cue).ToArray());
    }
}

/// <summary>Antonyms: a word → its opposite ("up" → "down"). Symmetric 1:1, taught both directions. The substrate's
/// dialectical core (meaning by negation). Source: GenesisNova GymLanguageFacts.AntonymPairs.</summary>
public sealed class Antonym : ISkill
{
    private static readonly (string, string)[] Pairs =
    {
        ("up","down"),("left","right"),("day","night"),("open","shut"),("win","lose"),("buy","sell"),
        ("push","pull"),("give","take"),("love","hate"),("war","peace"),("north","south"),("east","west"),
        ("king","queen"),("man","woman"),("boy","girl"),("front","back"),("top","bottom"),("question","answer"),
        ("friend","enemy"),("rise","fall"),("accept","reject"),("arrive","depart"),("remember","forget"),
        ("enter","exit"),("expand","shrink"),("sink","float"),("laugh","cry"),("import","export"),
    };
    // Directed both ways so opposite(a)=b AND opposite(b)=a.
    private static readonly (string cue, string ans)[] Directed =
        Pairs.SelectMany(p => new[] { (p.Item1, p.Item2), (p.Item2, p.Item1) }).ToArray();
    private static readonly string[] Frames = { "the opposite of {0}", "the antonym of {0}", "what is the opposite of {0}" };
    public string Name => "antonym";
    private (string cue, string ans) Draw(Random rng, int level) => Directed[rng.Next(Math.Min(Directed.Length, 20 + level * 8))];
    public Example Train(Random rng, int level) { var (c, a) = Draw(rng, level); return new(string.Format(Frames[rng.Next(Frames.Length)], c), a); }
    public Probe Probe(Random rng, int level) { var (c, a) = Draw(rng, level); return new($"the opposite of {c}", new[] { a }); }
}

/// <summary>Part → whole (meronymy): a part → the thing it belongs to ("wheel" → "car"). Distinct from is-a (a wheel
/// is not a KIND of car). Source: GenesisNova GymLanguageFacts.PartWhole.</summary>
public sealed class PartWhole : ISkill
{
    private static readonly (string Part, string Whole)[] Pairs =
    {
        ("wheel","car"),("engine","car"),("petal","flower"),("stem","flower"),("page","book"),("cover","book"),
        ("branch","tree"),("root","tree"),("bark","tree"),("wing","airplane"),("propeller","airplane"),
        ("fin","fish"),("gill","fish"),("mane","lion"),("hoof","horse"),("beak","eagle"),("feather","eagle"),
        ("sail","boat"),("anchor","boat"),("roof","house"),("chimney","house"),("brick","wall"),
        ("blade","knife"),("handle","hammer"),("keyboard","computer"),("screen","computer"),("pedal","bicycle"),
        ("spoke","bicycle"),("string","guitar"),("crust","pizza"),("yolk","egg"),("lace","shoe"),
    };
    private static readonly string[] Frames = { "what is a {0} part of", "a {0} is part of a", "what has a {0}" };
    public string Name => "partwhole";
    private (string part, string whole) Draw(Random rng, int level) => Pairs[rng.Next(Math.Min(Pairs.Length, 12 + level * 6))];
    public Example Train(Random rng, int level) { var (p, w) = Draw(rng, level); return new(string.Format(Frames[rng.Next(Frames.Length)], p), w); }
    public Probe Probe(Random rng, int level) { var (p, w) = Draw(rng, level); return new($"what is a {p} part of", new[] { w }); }
}

/// <summary>Ordered sequences: an element → its successor ("monday" → "tuesday"). A directed successor relation
/// (ordinal reasoning), not is-a or symmetric. Source: GenesisNova GymLanguageFacts.Sequences.</summary>
public sealed class WordSequence : ISkill
{
    private static readonly string[][] Series =
    {
        new[]{"monday","tuesday","wednesday","thursday","friday","saturday","sunday"},
        new[]{"january","february","march","april","may","june","july","august","september","october","november","december"},
        new[]{"spring","summer","autumn","winter"},
        new[]{"first","second","third","fourth","fifth","sixth","seventh","eighth"},
        new[]{"morning","noon","afternoon","evening"},
        new[]{"dawn","midday","dusk","midnight"},
    };
    private static readonly string[] Frames = { "what comes after {0}", "the next after {0} is", "after {0} comes" };
    public string Name => "sequence";
    private (string cur, string next) Draw(Random rng, int level)
    {
        var s = Series[rng.Next(Series.Length)];
        var i = rng.Next(s.Length - 1);
        return (s[i], s[i + 1]);
    }
    public Example Train(Random rng, int level) { var (c, n) = Draw(rng, level); return new(string.Format(Frames[rng.Next(Frames.Length)], c), n); }
    public Probe Probe(Random rng, int level) { var (c, n) = Draw(rng, level); return new($"what comes after {c}", new[] { n }); }
}

/// <summary>Numeric sequences: continue an arithmetic progression ("3 6 9" → "12"). The answer is a computed number
/// token, so it can't be memorised. Source: GenesisNova SequenceCreator (progression logic).</summary>
public sealed class NumberSequence : ISkill
{
    public string Name => "numseq";
    private (string display, string next) Draw(Random rng, int level)
    {
        var length = rng.Next(3, Math.Min(6, 3 + level) + 1);
        var step = rng.Next(1, Math.Max(2, 2 + level) + 1);
        var start = rng.Next(0, Math.Max(8, 8 + level * 3) + 1);
        var vals = Enumerable.Range(0, length).Select(j => start + j * step);
        return (string.Join(' ', vals), (start + length * step).ToString());
    }
    public Example Train(Random rng, int level) { var (d, n) = Draw(rng, level); return new($"next after {d} is", n); }
    public Probe Probe(Random rng, int level) { var (d, n) = Draw(rng, level); return new($"next after {d} is", new[] { n }); }
}

/// <summary>Arbitrary learned association: an entity → its fixed attribute ("otter" → "amber"). No prior can guess
/// it — pure associative recall, learnable only from the pairs. Source: GenesisNova AssociationRecallCreator.</summary>
public sealed class Association : ISkill
{
    // Each entity is bound to its OWN, SENSIBLE signature trait (a bijection — no two entities share a trait, and none
    // is nonsense like "walrus -> indigo"). Still pure associative recall — a char model has no prior linking walrus to
    // tusks, so it must LEARN the binding from the pairs — but now it reads correctly under every frame
    // ("the trait of walrus is tusks", "walrus goes with tusks", "describe walrus" -> "tusks").
    private static readonly (string Ent, string Attr)[] Bind =
    {
        ("otter","river"),  ("walrus","tusks"),   ("badger","stripes"), ("heron","marsh"),
        ("marten","branches"),("lynx","tufts"),   ("raven","wit"),      ("finch","seeds"),
        ("perch","scales"), ("gecko","grip"),     ("newt","ponds"),     ("toad","warts"),
        ("hare","speed"),   ("mole","tunnels"),   ("vole","grass"),     ("stoat","ermine"),
        ("weasel","cunning"),("ferret","burrow"), ("beaver","dam"),     ("marmot","whistle"),
        ("bison","horns"),  ("moose","antlers"),  ("quail","covey"),    ("crane","grace"),
    };
    private static readonly string[] Frames = { "the trait of {0} is", "{0} goes with", "{0} pairs with", "describe {0}" };
    public string Name => "assoc";
    private (string ent, string attr) Draw(Random rng, int level)
    {
        var n = level switch { 0 => 12, 1 => 18, _ => Bind.Length };
        return Bind[rng.Next(Math.Min(n, Bind.Length))];
    }
    public Example Train(Random rng, int level) { var (e, a) = Draw(rng, level); return new(string.Format(Frames[rng.Next(Frames.Length)], e), a); }
    public Probe Probe(Random rng, int level) { var (e, a) = Draw(rng, level); return new($"the trait of {e} is", new[] { a }); }
}

/// <summary>World-capital facts: country → its capital ("france" → "paris"). Clean single-token factual retrieval.
/// Source: GenesisNova LanguageDefaults.Facts (capitals subset; multi-token prose facts were skipped).</summary>
public sealed class Capital : ISkill
{
    private static readonly (string Country, string City)[] Table =
    {
        ("france","paris"),("england","london"),("germany","berlin"),("japan","tokyo"),("spain","madrid"),
        ("italy","rome"),("portugal","lisbon"),("austria","vienna"),("greece","athens"),("norway","oslo"),
        ("ireland","dublin"),("egypt","cairo"),("canada","ottawa"),("russia","moscow"),("china","beijing"),
        ("thailand","bangkok"),("cuba","havana"),("peru","lima"),("poland","warsaw"),("sweden","stockholm"),
        ("finland","helsinki"),("hungary","budapest"),("denmark","copenhagen"),("turkey","ankara"),
    };
    private static readonly string[] Frames = { "the capital of {0} is", "what is the capital of {0}" };
    public string Name => "capital";
    private (string c, string city) Draw(Random rng, int level) => Table[rng.Next(Math.Min(Table.Length, 10 + level * 6))];
    public Example Train(Random rng, int level) { var (c, city) = Draw(rng, level); return new(string.Format(Frames[rng.Next(Frames.Length)], c), city); }
    public Probe Probe(Random rng, int level) { var (c, city) = Draw(rng, level); return new($"the capital of {c} is", new[] { city }); }
}

/// <summary>Cloze / next-N continuation — the LLM-style skill. It trains next-token over a small self-contained corpus
/// of short sentences, and at probe time HIDES the last N tokens where N = level: the model must autoregressively
/// generate all N correctly (the gym grades the whole generated span). Difficulty PROGRESSES by how many tokens it has
/// to predict in a row, not by vocabulary — so it hardens toward free-running language modelling. Sentences start with
/// a distinct subject so a short prefix still determines a learnable continuation.</summary>
public static class SkillSet
{
    public static IReadOnlyList<ISkill> Default() => new ISkill[]
    {
        new Arithmetic('+'), new Arithmetic('-'), new Arithmetic('*'), new Arithmetic('/'),
        new NumberWords(), new Copy(),
        // ported from GenesisNova data creators / gym language facts (self-contained):
        new Category(), new Synonym(), new Antonym(), new PartWhole(),
        new WordSequence(), new NumberSequence(), new Association(), new Capital(),
    };
}
