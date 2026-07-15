using System.Text;
using PrismFormer;

// ═══════════════════════════════════════════════════════════════════════════════════════════════════════════
//  PairTest — the STUDIO.md data plane: a CORPUS lane (corpus/ *.txt) and a PAIR lane (pairs/ + gossip/ *.tsv) —
//  used differently, unified into one model. Gossip inbox dedups + caps. `prismnet pairtest`.
// ═══════════════════════════════════════════════════════════════════════════════════════════════════════════

static class PairTest
{
    const int CTX = 24;

    public static void Run()
    {
        Console.WriteLine("=== pair data-plane: corpus/ (fluency) + pairs/ + gossip/ (behaviour) -> one model (see STUDIO.md) ===\n");
        var root = Path.Combine(Path.GetTempPath(), "prism-pairtest-" + Guid.NewGuid().ToString("N")[..8]);
        var corpusDir = Path.Combine(root, "corpus"); var pairsDir = Path.Combine(root, "pairs"); var gossipDir = Path.Combine(root, "gossip");
        Directory.CreateDirectory(corpusDir); Directory.CreateDirectory(pairsDir);
        var v = new CharVocab();

        // CORPUS lane — continuous text, dense next-char windows (fluency)
        File.WriteAllText(Path.Combine(corpusDir, "prose.txt"), string.Concat(Enumerable.Repeat("the quick brown fox jumps over the lazy dog. ", 25)));
        // PAIR lane — independent prompt\ttarget examples (behaviour); target-only supervision, no cross-pair windows
        File.WriteAllLines(Path.Combine(pairsDir, "qa.tsv"), Enumerable.Repeat("what is the capital of france? \tparis", 25));

        var corpus = CorpusSource.FromFolders(CTX, v, corpusDir);
        var pairs = PairSource.FromFolders(CTX, v, pairsDir, gossipDir);
        var mix = new MixSource(corpus, pairs);
        Console.WriteLine($"corpus windows {corpus.Count:N0}  |  pair examples {pairs.Count:N0}  |  mixed {mix.Count:N0}   (vocab {v.Size}, ctx {CTX})");

        var m = new AlgFormer(v.Size, shifts: 24, layers: 3, maxContext: CTX, dModel: 96, frozenPrefix: 0, embedSeed: null, seed: 1);
        var loss0 = AvgLoss(m, mix, 400);
        Train(m, mix, epochs: 12, batch: 64, lr: 5e-3);
        Console.WriteLine($"trained on the mix        : avg loss {loss0:F3} -> {AvgLoss(m, mix, 400):F3}");
        Console.WriteLine($"  corpus fluency : \"the quick \"                    -> \"{Generate(m, v, "the quick ", 22)}\"");
        Console.WriteLine($"  pair behaviour : \"what is the capital of france? \" -> \"{Generate(m, v, "what is the capital of france? ", 6)}\"\n");

        // gossip inbox — neighbours append distilled pairs into gossip/ (dedup + cap)
        var inbox = new GossipInbox(gossipDir, cap: 1000);
        var fromA = new[] { ("the moon orbits the ", "earth"), ("water flows ", "downhill"), ("the sun rises in the ", "east") };
        var n1 = inbox.Add(fromA, from: "peer-A");
        var n2 = inbox.Add(fromA, from: "peer-B");
        var n3 = inbox.Add(Enumerable.Range(0, 3000).Select(i => ($"filler {i} is ", "noise")));
        Console.WriteLine("gossip inbox (neighbours writing distilled pairs into gossip/):");
        Console.WriteLine($"  peer-A sent 3 novel pairs      : +{n1}");
        Console.WriteLine($"  peer-B resent the same 3       : +{n2} (deduped)");
        Console.WriteLine($"  a flood of 3,000 pairs         : +{n3}, inbox held to cap = {inbox.Count:N0}");

        var pairsAfter = PairSource.FromFolders(CTX, v, pairsDir, gossipDir);
        Console.WriteLine($"\npair lane after gossip: {pairsAfter.Count:N0} examples (was {pairs.Count:N0}; gossip bled in {pairsAfter.Count - pairs.Count:N0} more)");
        Console.WriteLine("=> corpus and pairs stay separate lanes (windowed differently) but feed ONE model. Bleed = writing pairs into gossip/.");

        try { Directory.Delete(root, true); } catch { }
    }

    static double AvgLoss(AlgFormer m, IJobSource src, int n)
    {
        var g = m.NewGrads(); double s = 0; var step = Math.Max(1, src.Count / n); var cnt = 0;
        for (long i = 0; i + step <= src.Count && cnt < n; i += step) { var (c, t) = src.GetExample(i); s += m.Accumulate(c, t, g); cnt++; }
        return s / Math.Max(1, cnt);
    }
    static void Train(AlgFormer m, IJobSource src, int epochs, int batch, double lr)
    {
        var rng = new Random(7); var n = (int)src.Count;
        for (var ep = 0; ep < epochs; ep++)
        {
            var order = Enumerable.Range(0, n).OrderBy(_ => rng.Next()).ToArray();
            for (var s = 0; s < n; s += batch)
            {
                var g = m.NewGrads(); var cnt = 0;
                for (var i = s; i < s + batch && i < n; i++) { var (c, t) = src.GetExample(order[i]); m.Accumulate(c, t, g); cnt++; }
                m.Step(g, lr, scale: cnt);
            }
        }
    }
    static string Generate(AlgFormer m, CharVocab v, string prompt, int n)
    {
        var ids = v.Encode(prompt).ToList(); var sb = new StringBuilder();
        for (var k = 0; k < n; k++)
        {
            var ctx = new int[CTX]; var start = ids.Count - CTX;
            for (var i = 0; i < CTX; i++) { var j = start + i; ctx[i] = j >= 0 ? ids[j] : CharVocab.Pad; }
            var nx = m.Predict(ctx); ids.Add(nx); sb.Append(v.Chr(nx));
        }
        return sb.ToString();
    }
}
