// Copyright (c) 2026 Dongyang Stephen Chen, trading as Evaluated Applications. All rights reserved.
// Source-available for study and verification; commercial use, redistribution, or derivatives require permission. See LICENSE.

using System.Diagnostics;
using PrismFormer;

namespace PrismFormer.Gpu;

/// <summary>
/// IN-CONTEXT RECALL sizing sweep, trained on the GPU — the "remember people's names / what they're talking about" test.
/// A dialogue lists K (entity, topic) pairs, then queries one topic and must return the ENTITY that had it:
/// "alice tea . bob milk . carol juice . milk ? -> bob". Entities and topics are RE-PAIRED RANDOMLY every example, so the
/// association is NOT memorisable — the ONLY way to answer is to retrieve it from THIS context (pure in-context lookup,
/// the attention/induction mechanism). Held-out = unseen arrangements. Sweeps K (people to track) / DEPTH / WIDTH /
/// CONTEXT to find the smallest config that reliably remembers, and prints param count + fit for a 6 GB card.
/// </summary>
public static class GpuRecall
{
    static readonly Dictionary<string, int> _id = new(StringComparer.Ordinal);
    static readonly List<string> _words = new();
    static int Id(string w) { if (_id.TryGetValue(w, out var i)) return i; i = _id.Count; _id[w] = i; _words.Add(w); return i; }
    static int[] _ent = null!, _top = null!;
    static int DOT, Q, _vocab;
    const int E = 40, TN = 40;   // pools of distinct entity / topic symbols (chosen fresh + shuffled per example)

    public static void Run(int passes = 80)
    {
        try { Console.OutputEncoding = System.Text.Encoding.UTF8; } catch { }
        _ent = new int[E]; for (var i = 0; i < E; i++) _ent[i] = Id($"name{i}");
        _top = new int[TN]; for (var i = 0; i < TN; i++) _top[i] = Id($"topic{i}");
        DOT = Id("."); Q = Id("?");
        _vocab = _id.Count + 4;

        Console.WriteLine($"IN-CONTEXT RECALL on GPU — {GpuDevice.Describe}  (HasGpu={GpuDevice.HasGpu})");
        Console.WriteLine($"  \"name topic . name topic . … topic ? -> name\", pairs RE-PAIRED per example (not memorisable), held-out = unseen  ·  S=64");
        Console.WriteLine("  BEAT = held-out minus 1/K (a model that guesses a PRESENT name but not WHICH scores 1/K). >0 = real retrieval.\n");

        var jobs = new List<(string sweep, string name, int L, int dm, int K)>();
        foreach (var K in new[] { 2, 4, 8, 16 }) jobs.Add(("PEOPLE(K)", $"K{K}", 4, 128, K));
        foreach (var L in new[] { 1, 2, 4, 6 }) jobs.Add(("DEPTH", $"L{L}", L, 128, 8));
        foreach (var dm in new[] { 64, 128, 256 }) jobs.Add(("WIDTH", $"d{dm}", 4, dm, 8));

        var sw = Stopwatch.StartNew();
        string? last = null;
        foreach (var j in jobs)
        {
            if (j.sweep != last) { Console.WriteLine($"── {j.sweep} ──"); last = j.sweep; }
            var c = 3 * j.K + 4;
            var (tr, te, prm, curve) = TrainEval(j.L, j.dm, c, j.K, passes);
            var beat = te - 1.0 / j.K;
            Console.WriteLine($"   {j.name,-5} {prm / 1000.0,6:F0}k  train {tr,4:P0}  RECALL {te,4:P0}  1/K {1.0 / j.K,4:P0}  BEAT {beat,+5:P0}  ({sw.Elapsed.TotalSeconds,4:F0}s)  curve: {curve}");
        }
        Console.WriteLine($"\ndone in {sw.Elapsed.TotalSeconds:F0}s. RECALL near 100% = it reliably remembers who-said-what in context. Depth floor for recall + the K it holds = the reset's memory spec.");
        GpuDevice.Shutdown();
    }

    static (int[] ctx, int tgt) Gen(Random r, int K)
    {
        int[] Pick(int n, int pool) { var a = Enumerable.Range(0, pool).ToArray(); for (var i = pool - 1; i > 0; i--) { var j = r.Next(i + 1); (a[i], a[j]) = (a[j], a[i]); } return a[..n]; }
        var es = Pick(K, E); var ts = Pick(K, TN);
        var seq = new List<int>(3 * K + 2);
        for (var i = 0; i < K; i++) { seq.Add(_ent[es[i]]); seq.Add(_top[ts[i]]); seq.Add(DOT); }
        var q = r.Next(K);
        seq.Add(_top[ts[q]]); seq.Add(Q);
        return (seq.ToArray(), _ent[es[q]]);
    }

    static (List<(int[] ctx, int tgt)> train, List<(int[] ctx, int tgt)> test) Data(int K, int nTrain, int nTest)
    {
        var seen = new HashSet<string>(StringComparer.Ordinal);
        var train = new List<(int[], int)>(nTrain); var test = new List<(int[], int)>(nTest);
        var r = new Random(12345); var guard = 0;
        while ((train.Count < nTrain || test.Count < nTest) && guard++ < (nTrain + nTest) * 60)
        {
            var (ctx, tgt) = Gen(r, K);
            var key = string.Join(',', ctx);
            if (!seen.Add(key)) continue;
            uint h = 2166136261; foreach (var t in ctx) h = (h ^ (uint)t) * 16777619;
            if (h % 5 == 0) { if (test.Count < nTest) test.Add((ctx, tgt)); }
            else { if (train.Count < nTrain) train.Add((ctx, tgt)); }
        }
        return (train, test);
    }

    static (double tr, double te, long prm, string curve) TrainEval(int L, int dm, int c, int K, int passes)
    {
        var (train, test) = Data(K, 1500, 500);
        double[] Seed(int w)
        {
            var f = w < _words.Count ? PhasorCodec.Encode(_words[w]) : new double[PhasorCodec.Dim];
            if (dm >= f.Length) return f;
            var s = new double[dm]; Array.Copy(f, s, dm); return s;
        }
        var frozen = Math.Min(PhasorCodec.FrozenReals, dm);
        var cpu = new AlgFormer(_vocab, shifts: 64, layers: L, maxContext: c, dModel: dm, frozenPrefix: frozen, embedSeed: Seed, seed: 1);
        using var gt = new GpuTrainer(cpu);

        int[] Fit(int[] x) => x.Length > c ? x[^c..] : x;
        double Eval(List<(int[] ctx, int tgt)> d) { var ok = 0; foreach (var (ctx, tgt) in d) if (cpu.Predict(Fit(ctx)) == tgt) ok++; return ok / (double)d.Count; }

        var trainProbe = train.GetRange(0, Math.Min(500, train.Count));
        double bestTe = 0, bestTr = 0;
        var curve = new List<int>();
        var rng = new Random(1);
        for (var ep = 1; ep <= passes; ep++)
        {
            var idx = Enumerable.Range(0, train.Count).OrderBy(_ => rng.Next()).ToArray();
            var lr = 2e-3 * (1.0 - 0.9 * (ep - 1) / Math.Max(1, passes - 1));
            for (var b = 0; b < idx.Length; b += 128)
            {
                var batch = new List<(int[], int)>();
                for (var i = b; i < Math.Min(b + 128, idx.Length); i++) { var (ctx, tgt) = train[idx[i]]; batch.Add((Fit(ctx), tgt)); }
                gt.TrainBatch(batch, lr);
            }
            if (ep % 10 == 0 || ep == passes)
            {
                var tr = Eval(trainProbe); var te = Eval(test);
                if (te > bestTe) bestTe = te;
                if (tr > bestTr) bestTr = tr;
                curve.Add((int)Math.Round(te * 100));
                if (tr >= 0.99 && te >= 0.98) break;
            }
        }
        return (bestTr, bestTe, cpu.ParamCount, string.Join("→", curve));
    }
}
