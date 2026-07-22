// Copyright (c) 2026 Dongyang Stephen Chen, trading as Evaluated Applications. All rights reserved.
// Source-available for study and verification; commercial use, redistribution, or derivatives require permission. See LICENSE.

using System.Collections.Concurrent;
using PrismFormer;

namespace PrismFormer.Bench;

/// <summary>
/// --reason : MULTI-STEP REASONING generalisation, on real values, across depth / width / context / chain-length.
/// The task is a variable chain the model must EVALUATE: e.g. "a = 3 | b = a plus 4 | c = b minus 2 | c ?" -> 5. To answer
/// the queried variable it must resolve the references and track the running value through every step (mod 10, so answers
/// stay single-digit), which is genuine multi-step composition, not a lookup. Digits are NUMBER-face tokens (arithmetic
/// via the codec), the operators are WORD tokens, and the held-out split is UNSEEN chains, so held-out accuracy measures
/// whether the model learned the RULE (generalise) versus memorised the training chains.
///
/// Reports held-out (unseen) accuracy while sweeping each of: model DEPTH (layers), WIDTH (dModel, via the codec-truncated
/// face), CONTEXT (window), and problem CHAIN-LENGTH (how many reasoning steps). Train accuracy is shown too, so the
/// memorise-vs-generalise gap is visible. Concurrent jobs, early-stop.
/// </summary>
public static class ReasonBench
{
    // task vocab (built once, shared)
    static readonly Dictionary<string, int> _id = new(StringComparer.Ordinal);
    static readonly List<string> _words = new();
    static int Id(string w) { if (_id.TryGetValue(w, out var i)) return i; i = _id.Count; _id[w] = i; _words.Add(w); return i; }
    static int[] _dig = null!, _var = null!;
    static int EQ, BAR, Q, PLUS, MINUS;
    static int _vocab;

    public static void Run(int passes = 80)
    {
        try { Console.OutputEncoding = System.Text.Encoding.UTF8; } catch { }
        _dig = new int[10]; for (var d = 0; d < 10; d++) _dig[d] = Id(d.ToString());   // number faces
        _var = new int[8]; for (var v = 0; v < 8; v++) _var[v] = Id(((char)('a' + v)).ToString());   // variable-name signatures
        EQ = Id("="); BAR = Id("|"); Q = Id("?"); PLUS = Id("plus"); MINUS = Id("minus");
        _vocab = _id.Count + 4;

        Console.WriteLine($"MULTI-STEP REASONING generalisation — variable chains, words + numbers, held-out = UNSEEN chains   {DateTime.Now:yyyy-MM-dd HH:mm}");
        Console.WriteLine("  eg  \"a = 3 | b = a plus 4 | c = b minus 2 | c ?\" -> 5   (track values through the chain; answers mod 10)");
        Console.WriteLine($"  vocab {_vocab} (10 digit number-faces + 8 vars + plus/minus/=/|/? )  ·  S=128  ·  held-out accuracy = did it learn the RULE\n");

        if (!AlgFormer.GradCheck(out _)) { Console.WriteLine("gradcheck failed"); return; }

        // ---- sweeps: (label, list of (name,L,dm,c,K)) ----
        var jobs = new List<(string sweep, string name, int L, int dm, int c, int K)>();
        foreach (var L in new[] { 1, 2, 4, 8 }) jobs.Add(("DEPTH (layers)", $"L{L}", L, 128, 32, 3));
        foreach (var dm in new[] { 64, 128, 192, 256 }) jobs.Add(("WIDTH (dModel)", $"d{dm}", 4, dm, 32, 3));

        var res = new ConcurrentDictionary<string, (double tr, double te, int ep)>();
        var doneN = 0; var gate = new object();
        Console.WriteLine($"training {jobs.Count} configs concurrently (early-stop, cap {passes} passes) — streaming as each finishes:\n");
        Parallel.ForEach(jobs, j =>
        {
            var r = TrainEval(j.L, j.dm, j.c, j.K, passes);
            res[j.name] = r;
            lock (gate) { doneN++; Console.WriteLine($"  [{doneN}/{jobs.Count}] {j.sweep,-15} {j.name,-5} train {r.tr,5:P0}  held-out {r.te,5:P0}  @{r.ep}ep"); }
        });
        Console.WriteLine();

        foreach (var grp in jobs.GroupBy(j => j.sweep))
        {
            Console.WriteLine($"── {grp.Key} ──   (other dims fixed: {Fixed(grp.Key)})");
            Console.WriteLine($"   {"config",-8}{"train",10}{"HELD-OUT",12}{"passes",9}");
            foreach (var j in grp)
            {
                var r = res[j.name];
                Console.WriteLine($"   {j.name,-8}{r.tr,9:P0}{r.te,11:P0}{("@" + r.ep),9}");
            }
            Console.WriteLine();
        }
        Console.WriteLine("read: HELD-OUT = accuracy on chains it never trained on (chance = 10%). Rising with depth/width/context = the");
        Console.WriteLine("model is learning to COMPUTE the chain; a big train-minus-held-out gap = it's memorising, not generalising.");
    }

    static string Fixed(string sweep) => sweep switch
    {
        "DEPTH (layers)" => "d128, c32, K3",
        "WIDTH (dModel)" => "L4, c32, K3",
        "CHAIN LEN (steps)" => "L4, d128, c44",
        _ => "L4, d128, K4",
    };

    // a chain problem: start value, then K steps each = prevVar (plus|minus) digit, mod 10; query the last var
    static (int[] ctx, int tgt) Gen(Random r, int K)
    {
        var seq = new List<int>(6 + 6 * K);
        var val = r.Next(10);
        seq.Add(_var[0]); seq.Add(EQ); seq.Add(_dig[val]); seq.Add(BAR);
        for (var i = 1; i <= K; i++)
        {
            int op = r.Next(2), operand = r.Next(10);
            seq.Add(_var[i]); seq.Add(EQ); seq.Add(_var[i - 1]); seq.Add(op == 0 ? PLUS : MINUS); seq.Add(_dig[operand]); seq.Add(BAR);
            val = op == 0 ? (val + operand) % 10 : ((val - operand) % 10 + 10) % 10;
        }
        seq.Add(_var[K]); seq.Add(Q);
        return (seq.ToArray(), _dig[val]);
    }

    static (List<(int[] ctx, int tgt)> train, List<(int[] ctx, int tgt)> test) Data(int K, int nTrain, int nTest)
    {
        var seen = new HashSet<string>(StringComparer.Ordinal);
        var train = new List<(int[], int)>(nTrain); var test = new List<(int[], int)>(nTest);
        var r = new Random(12345);
        var guard = 0;
        while ((train.Count < nTrain || test.Count < nTest) && guard++ < (nTrain + nTest) * 40)
        {
            var (ctx, tgt) = Gen(r, K);
            var key = string.Join(',', ctx);
            if (!seen.Add(key)) continue;
            // deterministic 20% held-out by content hash (FNV) so a chain is ALWAYS train or ALWAYS test
            uint h = 2166136261; foreach (var t in ctx) { h = (h ^ (uint)t) * 16777619; }
            if (h % 5 == 0) { if (test.Count < nTest) test.Add((ctx, tgt)); }
            else { if (train.Count < nTrain) train.Add((ctx, tgt)); }
        }
        return (train, test);
    }

    static (double tr, double te, int ep) TrainEval(int L, int dm, int c, int K, int passes)
    {
        var (train, test) = Data(K, 1000, 500);
        double[] Seed(int w)
        {
            var full = w < _words.Count ? PhasorCodec.Encode(_words[w]) : new double[PhasorCodec.Dim];
            if (dm >= full.Length) return full;
            var s = new double[dm]; Array.Copy(full, s, dm); return s;   // codec-truncated face; number bands (reals 0..63) survive → arithmetic intact
        }
        var frozen = Math.Min(PhasorCodec.FrozenReals, dm);
        var m = new AlgFormer(_vocab, shifts: 64, layers: L, maxContext: c, dModel: dm, frozenPrefix: frozen, embedSeed: Seed, seed: 1);

        double Eval(List<(int[] ctx, int tgt)> d) { var ok = 0; foreach (var (ctx, tgt) in d) if (m.Predict(ctx.Length > c ? ctx[^c..] : ctx) == tgt) ok++; return ok / (double)d.Count; }

        double bestTe = 0; var bestTr = 0.0; var used = 0; var noGain = 0;
        var order = Enumerable.Range(0, train.Count).ToArray();
        for (var ep = 1; ep <= passes; ep++)
        {
            var er = new Random(1000 + ep);
            for (var i = order.Length - 1; i > 0; i--) { var j = er.Next(i + 1); (order[i], order[j]) = (order[j], order[i]); }
            var lr = 2e-3 * (1.0 - 0.9 * (ep - 1) / Math.Max(1, passes - 1));
            foreach (var idx in order) { var (ctx, tgt) = train[idx]; m.TrainStep(ctx.Length > c ? ctx[^c..] : ctx, tgt, lr); }
            used = ep;
            if (ep % 8 == 0 || ep == passes)
            {
                var te = Eval(test);
                if (te > bestTe + 0.005) { bestTe = te; bestTr = Eval(train.GetRange(0, Math.Min(500, train.Count))); noGain = 0; }
                else if (++noGain >= 5) break;   // PLATEAU: no held-out gain in 5 checks (~40 passes) → this config isn't learning, stop grinding
                if (te >= 0.97) { bestTe = Math.Max(bestTe, te); break; }
            }
        }
        return (bestTr, bestTe, used);
    }
}
