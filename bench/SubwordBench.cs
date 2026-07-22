// Copyright (c) 2026 Dongyang Stephen Chen, trading as Evaluated Applications. All rights reserved.
// Source-available for study and verification; commercial use, redistribution, or derivatives require permission. See LICENSE.

using PrismFormer;

namespace PrismFormer.Bench;

/// <summary>
/// --subwords : generate the deterministic subword list (n-grams of len 2..4) from the BabyLM corpus and SELF-VERIFY
/// the tokenizer (round-trip, greedy longest-match, determinism, digit number-faces, char-superset, empty-list
/// equivalence). Writes the ordered list to --out. Offline "build the vocab" step; wiring it into the live spec is
/// a separate, deliberate change.
///
/// Usage: prismformer-bench --subwords [--bi 2000] [--tri 1500] [--quad 500] [--corpus &lt;dir&gt;] [--out &lt;file&gt;]
/// </summary>
public static class SubwordBench
{
    public static void Run(string[] args)
    {
        try { Console.OutputEncoding = System.Text.Encoding.UTF8; } catch { }
        var bi = ArgInt(args, "--bi", 2000); var tri = ArgInt(args, "--tri", 1500); var quad = ArgInt(args, "--quad", 500);
        var root = FindRoot();
        var corpus = ArgStr(args, "--corpus", Path.Combine(root, "studio", "PrismStudio", "data", "text"));
        var outp = ArgStr(args, "--out", Path.Combine(root, "artifacts", "subwords-babylm.txt"));

        Console.WriteLine($"SUBWORD VOCAB BUILDER   {DateTime.Now:yyyy-MM-dd HH:mm}");
        Console.WriteLine($"corpus: {corpus}");
        if (!Directory.Exists(corpus)) { Console.WriteLine("  corpus dir not found — pass --corpus <dir of .txt files>"); return; }
        var files = Directory.GetFiles(corpus, "*.txt");
        Console.WriteLine($"  {files.Length} files, budget bi={bi} tri={tri} quad={quad}");
        if (files.Length == 0) { Console.WriteLine("  no .txt files"); return; }

        IEnumerable<string> Segments() { foreach (var f in files) yield return File.ReadAllText(f); }
        long chars = 0; foreach (var f in files) chars += new FileInfo(f).Length;

        // NUMBER-CLEAN vocab so the single-digit arithmetic scratchpad tokenises consistently and the column algorithm can
        // extrapolate. Keep only: pure-TEXT subwords, and pure-digit numbers of AT MOST 2 digits ("00".."99", the whole-number
        // faces). DROP: mixed digit+text ("+4","=4","c0","1st" — they glue digits to operators/letters), and 3+ digit numbers
        // ("156","1990" — they're an inconsistent big chunk the column algorithm would have to re-split). Every number then
        // tokenises into uniform ≤2-digit chunks; every digit inside the scratchpad stays a clean single char.
        static bool KeepSub(string w)
        {
            var digit = w.Count(char.IsDigit);
            if (digit == 0) return true;                 // pure text — keep
            if (digit != w.Length) return false;         // mixed digit+text — drop
            return w.Length <= 2;                        // pure number — keep only 1..2 digits
        }
        static List<string> CleanDigits(List<string> l) => l.Where(KeepSub).ToList();

        var raw = SubwordBuilder.FromSegments(Segments(), bi, tri, quad);
        var list = CleanDigits(raw);
        Console.WriteLine($"  filtered {raw.Count - list.Count} mixed digit/text subwords (kept pure-digit + pure-text)");
        var v = new SubwordVocab(list);
        int L2 = list.Count(w => w.Length == 2), L3 = list.Count(w => w.Length == 3), L4 = list.Count(w => w.Length == 4);
        Console.WriteLine($"  corpus ~{chars / 1_000_000.0:F1} M chars  ->  {list.Count} subwords ({L2} bi, {L3} tri, {L4} quad)  |  total vocab {v.Size}");
        Console.WriteLine($"  embedding at d=256: {(long)v.Size * 256:N0} params (+{(long)list.Count * 256:N0} vs char-only)");
        Console.WriteLine($"  sample tri: {string.Join(" ", list.Where(w => w.Length == 3).Take(16).Select(Show))}");
        Console.WriteLine($"  sample quad: {string.Join(" ", list.Where(w => w.Length == 4).Take(12).Select(Show))}");

        // ---------------- self-verification ----------------
        var pass = 0; var fail = 0;
        void Check(string name, bool ok) { Console.WriteLine($"    [{(ok ? "PASS" : "FAIL")}] {name}"); if (ok) pass++; else fail++; }
        Console.WriteLine("verification:");

        var empty = new SubwordVocab(Array.Empty<string>());
        var cv = new CharVocab();
        var sample = "The 47 cats ran, but 3+4=7! Newlines\nand tabs\tfold.";
        var ce = cv.Encode(sample); var ee = empty.Encode(sample);
        Check("empty list == char vocab (ids + length)", ee.Length == ce.Length && ee.SequenceEqual(ce));

        var folded = new string(sample.Select(c => c >= CharVocab.Lo && c <= CharVocab.Hi ? c : ' ').ToArray());
        Check("Decode(Encode(text)) == folded text", v.Decode(v.Encode(sample)) == folded);

        // greedy longest-match: a quadgram (if any) tokenizes to ONE token, and prefers the longest unit
        var quadProbe = list.FirstOrDefault(w => w.Length == 4 && w.All(char.IsLetter));
        var quadOne = quadProbe != null && v.Encode(quadProbe).Length == 1;
        Check($"a known quadgram is ONE token (longest-match)  ({quadProbe ?? "none found"})", quadOne);

        var list2 = CleanDigits(SubwordBuilder.FromSegments(Segments(), bi, tri, quad));
        Check("regeneration is bit-identical (deterministic)", list2.SequenceEqual(list));

        var allDigits = Enumerable.Range(0, 100).All(n => list.Contains($"{n / 10}{n % 10}"));
        Check("all 100 two-digit pairs present", allDigits);

        var num47 = list.Contains("47") && v.Symbol(v.CharN + list.IndexOf("47")) == "47" && PhasorCodec.IsNumber("47", out var val) && val == 47;
        Check("digit token \"47\" -> number face (IsNumber, value 47)", num47);
        var num3 = list.FirstOrDefault(w => w.Length == 3 && w.All(char.IsDigit));
        if (num3 != null) Check($"a 3-digit token \"{num3}\" -> number face", PhasorCodec.IsNumber(num3, out var v3) && v3 == int.Parse(num3));

        var superset = Enumerable.Range(0, CharVocab.N).All(id => v.Symbol(id) == (id == CharVocab.End ? "\n" : cv.Chr(id).ToString()));
        Check("ids 0..95 identical to CharVocab (char rows carry over)", superset);

        Console.WriteLine($"  => {pass} passed, {fail} failed");

        Directory.CreateDirectory(Path.GetDirectoryName(outp)!);
        File.WriteAllLines(outp, list);
        Console.WriteLine($"wrote {list.Count} subwords -> {outp}");
        if (fail > 0) Console.WriteLine("VERIFICATION FAILED");
    }

    static string Show(string w) => w.Replace(" ", "_");
    static int ArgInt(string[] a, string k, int d) { for (var i = 0; i < a.Length - 1; i++) if (a[i] == k && int.TryParse(a[i + 1], out var v)) return v; return d; }
    static string ArgStr(string[] a, string k, string d) { for (var i = 0; i < a.Length - 1; i++) if (a[i] == k) return a[i + 1]; return d; }

    static string FindRoot()
    {
        var dir = AppContext.BaseDirectory;
        for (var i = 0; i < 8 && dir != null; i++)
        {
            if (Directory.Exists(Path.Combine(dir, "studio")) || Directory.GetFiles(dir, "*.slnx").Length > 0) return dir;
            dir = Directory.GetParent(dir)?.FullName;
        }
        return Directory.GetCurrentDirectory();
    }
}
