// Copyright (c) 2026 Dongyang Stephen Chen, trading as Evaluated Applications. All rights reserved.
// Source-available for study and verification; commercial use, redistribution, or derivatives require permission. See LICENSE.

using PrismFormer;

namespace PrismFormer.Bench;

/// <summary>
/// --bigrams : generate the deterministic bigram-subword list from the BabyLM corpus and SELF-VERIFY the tokenizer
/// (round-trip, greedy match, determinism, digit-bigram number-faces, char-superset, empty-list equivalence). Writes
/// the ordered list to the --out path. This is the offline "build the vocab" step; wiring it into the live spec
/// (PrismSpec/StudioModel/LoadUpgrade) is a separate, deliberate change.
///
/// Usage: prismformer-bench --bigrams [--topk 2000] [--corpus &lt;dir of *.txt&gt;] [--out &lt;file&gt;]
/// </summary>
public static class BigramBench
{
    public static void Run(string[] args)
    {
        try { Console.OutputEncoding = System.Text.Encoding.UTF8; } catch { }
        var topk = ArgInt(args, "--topk", 2000);
        var root = FindRoot();
        var corpus = ArgStr(args, "--corpus", Path.Combine(root, "studio", "PrismStudio", "data", "text"));
        var outp = ArgStr(args, "--out", Path.Combine(root, "artifacts", "bigrams-babylm.txt"));

        Console.WriteLine($"BIGRAM VOCAB BUILDER   {DateTime.Now:yyyy-MM-dd HH:mm}");
        Console.WriteLine($"corpus: {corpus}");
        if (!Directory.Exists(corpus)) { Console.WriteLine("  corpus dir not found — pass --corpus <dir of .txt files>"); return; }
        var files = Directory.GetFiles(corpus, "*.txt");
        Console.WriteLine($"  {files.Length} files, topk={topk}");
        if (files.Length == 0) { Console.WriteLine("  no .txt files"); return; }

        // each file is one segment (bigrams never span files); read lazily to keep memory flat
        IEnumerable<string> Segments() { foreach (var f in files) yield return File.ReadAllText(f); }
        long chars = 0; foreach (var f in files) chars += new FileInfo(f).Length;

        var list = BigramBuilder.FromSegments(Segments(), topk);
        var v = new BigramVocab(list);
        Console.WriteLine($"  corpus ~{chars / 1_000_000.0:F1} M chars  ->  {list.Count} bigrams  |  total vocab {v.Size} (chars {v.CharN} + bigrams {list.Count})");
        Console.WriteLine($"  embedding at d=256: {(long)v.Size * 256:N0} params (+{(long)list.Count * 256:N0} vs char-only)");
        Console.WriteLine($"  most-frequent-ish sample (canonical order): {string.Join(" ", list.Take(24).Select(Show))}");

        // ---------------- self-verification ----------------
        var pass = 0; var fail = 0;
        void Check(string name, bool ok) { Console.WriteLine($"    [{(ok ? "PASS" : "FAIL")}] {name}"); if (ok) pass++; else fail++; }
        Console.WriteLine("verification:");

        // 1. empty list == char-level, byte-identical ids and length
        var empty = new BigramVocab(Array.Empty<string>());
        var cv = new CharVocab();
        var sample = "The 47 cats ran, but 3+4=7! Newlines\nand tabs\tfold.";
        var ce = cv.Encode(sample); var ee = empty.Encode(sample);
        Check("empty bigram list == char vocab (ids + length)", ee.Length == ce.Length && ee.SequenceEqual(ce));

        // 2. round-trip through the bigram vocab reproduces the FOLDED text
        var folded = new string(sample.Select(c => c >= CharVocab.Lo && c <= CharVocab.Hi ? c : ' ').ToArray());
        Check("Decode(Encode(text)) == folded text", v.Decode(v.Encode(sample)) == folded);

        // 3. greedy match actually shortens the sequence where a known bigram exists
        var biErr = false; string? demo = null;
        var probe = list.FirstOrDefault(b => b[0] != ' ' && b[1] != ' ' && char.IsLetter(b[0]) && char.IsLetter(b[1]));
        if (probe != null) { var enc = v.Encode(probe); demo = $"'{probe}' -> {enc.Length} token(s)"; biErr = enc.Length != 1; }
        Check($"a known bigram tokenizes to ONE token  ({demo ?? "no letter bigram found"})", probe != null && !biErr);

        // 4. determinism: rebuild from the same segments → identical list
        var list2 = BigramBuilder.FromSegments(Segments(), topk);
        Check("regeneration is bit-identical (deterministic)", list2.SequenceEqual(list));

        // 5. digit bigrams present, addressable, and hit the NUMBER face
        var has47 = list.Contains("47");
        var id47 = has47 ? v.CharN + list.IndexOf("47") : -1;
        var num47 = has47 && v.Symbol(id47) == "47" && PhasorCodec.IsNumber("47", out var val) && val == 47;
        var allDigits = Enumerable.Range(0, 100).All(n => list.Contains($"{n / 10}{n % 10}"));
        Check("all 100 digit pairs present (full number permutation)", allDigits);
        Check("digit bigram \"47\" -> number face (IsNumber, value 47)", num47);

        // 6. char-superset: ids 0..95 decode exactly like CharVocab
        var superset = Enumerable.Range(0, CharVocab.N).All(id => v.Symbol(id) == (id == CharVocab.End ? "\n" : cv.Chr(id).ToString()));
        Check("ids 0..95 identical to CharVocab (superset → char rows carry over)", superset);

        Console.WriteLine($"  => {pass} passed, {fail} failed");

        Directory.CreateDirectory(Path.GetDirectoryName(outp)!);
        File.WriteAllLines(outp, list);
        Console.WriteLine($"wrote {list.Count} bigrams -> {outp}");
        if (fail > 0) Console.WriteLine("VERIFICATION FAILED");
    }

    static string Show(string b) => b.Replace(" ", "_");

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
