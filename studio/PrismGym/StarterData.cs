using PrismFormer;

namespace PrismGym;

/// <summary>
/// Seeds a data folder with STARTER training files (see STUDIO.md). The old live trainers become file producers:
/// the BabyLM corpus is written to <c>corpus/*.txt</c> (fluency), and the Gym skill generators are written to
/// <c>pairs/*.tsv</c> (behaviour, one <c>prompt\ttarget</c> line each). After this, training is just "train on the
/// folder" — and anyone can extend it by dropping more files, no code.
/// </summary>
public static class StarterData
{
    public static void GenerateAll(string dataDir, int corpusWords, Action<string> log)
    {
        GeneratePairs(dataDir, log);
        GenerateCorpus(dataDir, corpusWords, log);
        log("[starter] done — press Train.");
    }

    /// <summary>Install the SHIPPED starter files (pre-generated corpus chunks + pairs) into the data folder — no
    /// download, no code. Copies everything under <paramref name="shippedDir"/> into <paramref name="dataDir"/>,
    /// preserving the corpus/ and pairs/ layout. Existing user files with other names are left alone.</summary>
    public static void InstallShipped(string shippedDir, string dataDir, Action<string> log)
    {
        if (!Directory.Exists(shippedDir)) { log($"[starter] shipped data not found at {shippedDir}"); return; }
        var n = 0;
        foreach (var src in Directory.EnumerateFiles(shippedDir, "*", SearchOption.AllDirectories))
        {
            var dst = Path.Combine(dataDir, Path.GetRelativePath(shippedDir, src));
            Directory.CreateDirectory(Path.GetDirectoryName(dst)!);
            File.Copy(src, dst, overwrite: true); n++;
        }
        log($"[starter] installed {n} starter files into {dataDir} — press Train.");
    }

    /// <summary>Write the BabyLM corpus (word-budgeted) to <c>corpus/babylm/</c>, split into multiple files
    /// (~<paramref name="charsPerFile"/> each, cut at a space) so it's not one giant blob.</summary>
    public static void GenerateCorpus(string dataDir, int words, Action<string> log, int charsPerFile = 2_000_000)
    {
        var babyDir = Path.Combine(dataDir, "text");
        Directory.CreateDirectory(babyDir);
        foreach (var old in Directory.GetFiles(babyDir, "babylm*.txt")) File.Delete(old);   // clear stale chunks
        try
        {
            log($"[starter] loading BabyLM corpus ({words:N0} words)…");
            var text = new BabyLmRun { WordBudget = words }.CorpusText(log);
            if (text.Length == 0) { log("[starter] corpus empty (download failed?) — skipped"); return; }
            int part = 0, pos = 0;
            while (pos < text.Length)
            {
                var end = Math.Min(text.Length, pos + charsPerFile);
                if (end < text.Length)
                {
                    var sp = text.LastIndexOf(' ', end - 1, Math.Min(end - pos, 8000));   // back up to a space so a word isn't split
                    if (sp > pos) end = sp + 1;
                }
                File.WriteAllText(Path.Combine(babyDir, $"babylm-{++part:D3}.txt"), text[pos..end]);
                pos = end;
            }
            log($"[starter] wrote corpus/babylm/ — {part} files, {text.Length:N0} chars total");
        }
        catch (Exception e) { log($"[starter] corpus skipped ({e.Message.Split('\n')[0]})"); }
    }

    /// <summary>Run every Gym skill generator across a few difficulty levels and write <c>prompt\ttarget</c> pairs.</summary>
    public static void GeneratePairs(string dataDir, Action<string> log, int perLevel = 200, int levels = 3)
    {
        var pairsDir = Path.Combine(dataDir, "pairs"); Directory.CreateDirectory(pairsDir);
        var rng = new Random(1);
        foreach (var s in SkillSet.Default())
        {
            var seen = new HashSet<string>();
            for (var lvl = 1; lvl <= levels; lvl++)
                for (var n = 0; n < perLevel; n++)
                {
                    var ex = s.Train(rng, lvl);
                    var prompt = ex.Prompt.Trim(); var target = ex.Completion.Trim();
                    if (prompt.Length == 0 || target.Length == 0) continue;
                    seen.Add($"{prompt}\t{target}");
                }
            var file = Path.Combine(pairsDir, Safe(s.Name) + ".tsv");
            File.WriteAllLines(file, seen);
            log($"[starter] wrote pairs/{Safe(s.Name)}.tsv ({seen.Count} pairs)");
        }
    }

    static string Safe(string name) { foreach (var c in Path.GetInvalidFileNameChars()) name = name.Replace(c, '_'); return name; }
}
