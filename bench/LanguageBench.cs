// Copyright (c) 2026 Dongyang Stephen Chen, trading as Evaluated Applications. All rights reserved.
// Source-available for study and verification; commercial use, redistribution, or derivatives require permission. See LICENSE.

using PrismFormer;

namespace PrismFormer.Bench;

/// <summary>
/// Character-level LANGUAGE MODELLING in isolation (run with <c>--lm</c>): PrismFormer vs a parameter-matched dense
/// transformer predicting the next character of a small embedded public-domain text — the actual LLM objective, at a
/// size that trains in seconds (no base model to train). Held-out is a contiguous tail the model never trains on.
/// Metrics: next-char accuracy and bits/char (lower is better). The two models train side by side on separate threads.
/// </summary>
internal static class LanguageBench
{
    public static void Run(int epochs = 30, int seeds = 5)
    {
        try { Console.OutputEncoding = System.Text.Encoding.UTF8; } catch { }
        var text = Text;
        var chars = text.Distinct().OrderBy(c => c).ToArray();
        var cid = new Dictionary<char, int>(); for (var i = 0; i < chars.Length; i++) cid[chars[i]] = i;
        int V = chars.Length; const int ctx = 8;
        var split = (int)(text.Length * 0.85);
        List<(int[] Ctx, int Tgt)> Build(int lo, int hi)
        {
            var l = new List<(int[], int)>();
            for (var i = Math.Max(lo, ctx); i < hi; i++)
            {
                var c = new int[ctx];
                for (var k = 0; k < ctx; k++) c[k] = cid[text[i - ctx + k]];
                l.Add((c, cid[text[i]]));
            }
            return l;
        }
        var train = Build(0, split); var held = Build(split, text.Length);

        var vocab = Math.Max(64, V + 4);
        double[] Seed(int w) => w < V ? PhasorCodec.Encode(chars[w].ToString()) : new double[PhasorCodec.Dim];
        var probeAlg = AlgFormer.Mini(vocab, embedSeed: Seed, seed: 1);
        var targetParams = probeAlg.ParamCount;
        (int d, int L, int ff) best = (32, 2, 64); var bestDelta = long.MaxValue;
        foreach (var d in new[] { 24, 32, 40, 48, 56, 64, 80, 96 })
            foreach (var L in new[] { 2, 3, 4 })
                foreach (var ff in new[] { 32, 64, 96, 128, 192, 256 })
                { var p = new MiniTransformer(vocab, d, ff, L, ctx, 42); if (Math.Abs(p.ParamCount - targetParams) < bestDelta) { bestDelta = Math.Abs(p.ParamCount - targetParams); best = (d, L, ff); } }

        Console.WriteLine($"PrismFormer vs pound-for-pound transformer - CHARACTER LANGUAGE MODELLING (both temperature-calibrated)   {DateTime.Now:yyyy-MM-dd HH:mm}");
        if (!AlgFormer.GradCheck(out var ar) || !MiniTransformer.GradCheck(out var xr)) { Console.WriteLine("GRADCHECK FAILED"); return; }
        var modeChar = held.GroupBy(e => e.Tgt).OrderByDescending(g => g.Count()).First().Key;
        var baseAcc = held.Count(e => e.Tgt == modeChar) / (double)held.Count;
        Console.WriteLine($"gradchecks pass   corpus {text.Length} chars, vocab {V}, ctx {ctx}   train {train.Count} / held {held.Count}   {seeds} seeds × {epochs} epochs");
        Console.WriteLine($"params: transformer {new MiniTransformer(vocab, best.d, best.ff, best.L, ctx, 42).ParamCount:N0} (d={best.d}, dff={best.ff}, L={best.L})   prism {targetParams:N0}   baseline acc {baseAcc:P1}\n");

        // temperature scaling for BOTH (standard LM calibration): pick tau on a VAL slice, report bits/char on a TEST slice.
        var valN = Math.Max(20, held.Count / 3);
        var val = held.Take(valN).ToList(); var test = held.Skip(valN).ToList();
        double Bits(Func<int[], double[]> logits, List<(int[] Ctx, int Tgt)> set, double tau)
        {
            var s = 0.0;
            foreach (var e in set)
            {
                var lg = logits(e.Ctx);
                var mx = double.NegativeInfinity; for (var w = 0; w < lg.Length; w++) { var v = lg[w] / tau; if (v > mx) mx = v; }
                var sum = 0.0; for (var w = 0; w < lg.Length; w++) sum += Math.Exp(lg[w] / tau - mx);
                s += -(lg[e.Tgt] / tau - mx - Math.Log(sum));
            }
            return s / set.Count / Math.Log(2);
        }
        var taus = new[] { 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0 };
        (double acc, double bits) Report(Func<int[], double[]> logits, Func<int[], int> predict)
        {
            var tau = taus.OrderBy(t => Bits(logits, val, t)).First();
            return (test.Count(e => predict(e.Ctx) == e.Tgt) / (double)test.Count, Bits(logits, test, tau));
        }

        double[] pAcc = new double[seeds], pBits = new double[seeds], xAcc = new double[seeds], xBits = new double[seeds];
        Parallel.For(0, seeds, s =>
        {
            var alg = AlgFormer.Mini(vocab, embedSeed: Seed, seed: 1 + s);
            var xf = new MiniTransformer(vocab, best.d, best.ff, best.L, ctx, 42 + s);
            for (var ep = 1; ep <= epochs; ep++)
            {
                var lr = 2e-3 * (1.0 - 0.9 * (ep - 1) / Math.Max(1, epochs - 1));
                var order = Enumerable.Range(0, train.Count).ToArray();
                var er = new Random(1000 + ep + 97 * s);
                for (var i = order.Length - 1; i > 0; i--) { var j = er.Next(i + 1); (order[i], order[j]) = (order[j], order[i]); }
                foreach (var i in order) { var (c, t) = train[i]; alg.TrainStep(c, t, lr); }
                foreach (var i in order) { var (c, t) = train[i]; xf.TrainStep(c, t, lr); }
            }
            (pAcc[s], pBits[s]) = Report(alg.LogitsFor, alg.Predict);
            (xAcc[s], xBits[s]) = Report(xf.LogitsFor, xf.Predict);
        });

        (double m, double sd) MS(double[] a) { var m = a.Average(); return (m, a.Length > 1 ? Math.Sqrt(a.Sum(x => (x - m) * (x - m)) / (a.Length - 1)) : 0); }
        var (pa, pas) = MS(pAcc); var (pbi, pbis) = MS(pBits); var (xa, xas) = MS(xAcc); var (xbi, xbis) = MS(xBits);

        Console.WriteLine("\nheld-out language modelling (test slice, temperature-calibrated on val; mean ± sd) ---------");
        Console.WriteLine($"  {"model",-12} {"next-char acc",-18} bits/char (lower=better)");
        Console.WriteLine($"  {"transformer",-12} {$"{xa:P1} ± {xas:P1}",-18} {xbi:F3} ± {xbis:F3}");
        Console.WriteLine($"  {"prismformer",-12} {$"{pa:P1} ± {pas:P1}",-18} {pbi:F3} ± {pbis:F3}");
        Console.WriteLine($"  {"baseline",-12} {baseAcc,13:P1}       {Math.Log2(V):F3}  (unigram / uniform)");
    }

    // Embedded public-domain text (Lewis Carroll, Alice's Adventures in Wonderland, 1865) — self-contained, reproducible.
    private const string Text =
        "Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: " +
        "once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, " +
        "and what is the use of a book, thought Alice, without pictures or conversations? So she was considering in her own mind, " +
        "as well as she could, for the hot day made her feel very sleepy and stupid, whether the pleasure of making a daisy chain " +
        "would be worth the trouble of getting up and picking the daisies, when suddenly a White Rabbit with pink eyes ran close by her. " +
        "There was nothing so very remarkable in that; nor did Alice think it so very much out of the way to hear the Rabbit say to itself, " +
        "Oh dear! Oh dear! I shall be late! But when the Rabbit actually took a watch out of its waistcoat pocket, and looked at it, " +
        "and then hurried on, Alice started to her feet, for it flashed across her mind that she had never before seen a rabbit with " +
        "either a waistcoat pocket, or a watch to take out of it, and burning with curiosity, she ran across the field after it, " +
        "and fortunately was just in time to see it pop down a large rabbit hole under the hedge. In another moment down went Alice " +
        "after it, never once considering how in the world she was to get out again. The rabbit hole went straight on like a tunnel " +
        "for some way, and then dipped suddenly down, so suddenly that Alice had not a moment to think about stopping herself before " +
        "she found herself falling down a very deep well. Either the well was very deep, or she fell very slowly, for she had plenty " +
        "of time as she went down to look about her, and to wonder what was going to happen next.";
}
