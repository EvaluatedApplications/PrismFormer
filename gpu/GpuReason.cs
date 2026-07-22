// Copyright (c) 2026 Dongyang Stephen Chen, trading as Evaluated Applications. All rights reserved.
// Source-available for study and verification; commercial use, redistribution, or derivatives require permission. See LICENSE.

using System.Diagnostics;
using PrismFormer;

namespace PrismFormer.Gpu;

/// <summary>
/// MULTI-STEP REASONING generalisation, trained on the GPU (via <see cref="GpuTrainer"/>), used to SIZE the reset model:
/// bring the task into the learnable regime and read where more depth/width/context stops helping.
///
/// Task = additive variable chains the model must EVALUATE: "a = 3 | b = a plus 4 | c = b plus 2 | c ?" -> 9. The answer
/// is the REAL running sum (NOT mod 10), so the target is exactly the number-face the codec's bind=add produces natively
/// (aligned, not fighting the representation). Operands 0..9 and every reachable answer 0..maxSum are NUMBER-face tokens;
/// operators/vars/punctuation are symbol tokens. Held-out = UNSEEN chains, so held-out accuracy = did it learn the RULE.
/// We report a majority-class BASELINE per config (sums cluster, so "chance" is the mode's frequency, not 1/classes).
/// Sweeps CHAIN-LENGTH / DEPTH / WIDTH / CONTEXT, one card, sequential, but each config is minutes on the GPU.
/// </summary>
public static class GpuReason
{
    static readonly Dictionary<string, int> _id = new(StringComparer.Ordinal);
    static readonly List<string> _words = new();
    static int Id(string w) { if (_id.TryGetValue(w, out var i)) return i; i = _id.Count; _id[w] = i; _words.Add(w); return i; }
    static int[] _num = null!, _var = null!;   // _num[v] = token for the number-face of value v (operands AND answers)
    static int EQ, BAR, Q, PLUS, MINUS, _vocab, _maxAns;
    const int MaxK = 8;           // deepest chain in the sweep — fixes the answer range and the var-name set
    const bool UseMinus = false;  // addition-only keeps values non-negative and the task aligned with bind=add
    const bool QueryRandom = true;// query a RANDOM earlier variable (not always the last) → forces reference resolution, not "sum everything"

    public static void Run(int passes = 120)
    {
        try { Console.OutputEncoding = System.Text.Encoding.UTF8; } catch { }
        _maxAns = 9 * (MaxK + 1);                                             // biggest reachable additive sum
        _num = new int[_maxAns + 1]; for (var v = 0; v <= _maxAns; v++) _num[v] = Id(v.ToString());   // number faces 0..maxSum
        _var = new int[MaxK + 1]; for (var v = 0; v <= MaxK; v++) _var[v] = Id(((char)('a' + v)).ToString());
        EQ = Id("="); BAR = Id("|"); Q = Id("?"); PLUS = Id("plus"); MINUS = Id("minus");
        _vocab = _id.Count + 4;

        Console.WriteLine($"ADDITIVE REASONING on GPU — {GpuDevice.Describe}  (HasGpu={GpuDevice.HasGpu})");
        Console.WriteLine($"  real-sum target (aligned with bind=add), answers 0..{_maxAns}, query={(QueryRandom ? "RANDOM earlier var (reference resolution)" : "last var")}, held-out = UNSEEN  ·  S=64");
        Console.WriteLine("  BEAT-BASE = held-out minus majority-class baseline. RACE: tiny (L2/d64) vs bigger (L6/d160) across chain depth → where does depth EARN its keep?\n");

        var jobs = new List<(string sweep, string name, int L, int dm, int c, int K)>();
        foreach (var K in new[] { 3, 5, 8 })
        {
            var ctx = 8 + 6 * K;   // enough window for the whole chain + query
            jobs.Add(($"K{K}", "tiny", 2, 64, ctx, K));
            jobs.Add(($"K{K}", "big", 6, 160, ctx, K));
        }

        var sw = Stopwatch.StartNew();
        string? lastSweep = null;
        foreach (var j in jobs)
        {
            if (j.sweep != lastSweep) { Console.WriteLine($"── {j.sweep} ──"); lastSweep = j.sweep; }
            var (tr, te, baseline, curve) = TrainEval(j.L, j.dm, j.c, j.K, passes);
            var beat = te - baseline;
            Console.WriteLine($"   {j.name,-5}  train {tr,5:P0}   held-out {te,5:P0}   base {baseline,4:P0}   BEAT-BASE {beat,+5:P0}   ({sw.Elapsed.TotalSeconds,4:F0}s)   curve: {curve}");
        }
        Console.WriteLine($"\ndone in {sw.Elapsed.TotalSeconds:F0}s on GPU. Read: pick the SMALLEST config whose BEAT-BASE saturates — adding depth/width past that buys nothing.");
        GpuDevice.Shutdown();
    }

    // an additive chain: start 0..9, then K steps each = prevVar plus digit; query var (last, or a random earlier one);
    // answer = that var's REAL value. Random query forces the model to RESOLVE the reference, not just sum everything.
    static (int[] ctx, int tgt) Gen(Random r, int K)
    {
        var seq = new List<int>(6 + 6 * K);
        var vals = new int[K + 1];
        var val = r.Next(10); vals[0] = val;
        seq.Add(_var[0]); seq.Add(EQ); seq.Add(_num[val]); seq.Add(BAR);
        for (var i = 1; i <= K; i++)
        {
            var minus = UseMinus && r.Next(2) == 1 && val >= 9;
            var operand = minus ? r.Next(val + 1) : r.Next(10);
            seq.Add(_var[i]); seq.Add(EQ); seq.Add(_var[i - 1]); seq.Add(minus ? MINUS : PLUS); seq.Add(_num[operand]); seq.Add(BAR);
            val = minus ? val - operand : val + operand; vals[i] = val;
        }
        var qv = QueryRandom ? 1 + r.Next(K) : K;   // which variable is asked (1..K)
        seq.Add(_var[qv]); seq.Add(Q);
        return (seq.ToArray(), _num[vals[qv]]);
    }

    static (List<(int[] ctx, int tgt)> train, List<(int[] ctx, int tgt)> test) Data(int K, int nTrain, int nTest)
    {
        var seen = new HashSet<string>(StringComparer.Ordinal);
        var train = new List<(int[], int)>(nTrain); var test = new List<(int[], int)>(nTest);
        var r = new Random(12345); var guard = 0;
        while ((train.Count < nTrain || test.Count < nTest) && guard++ < (nTrain + nTest) * 200)
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

    static (double tr, double te, double baseline, string curve) TrainEval(int L, int dm, int c, int K, int passes)
    {
        var (train, test) = Data(K, 1500, 500);
        // majority-class baseline on the TEST split (additive sums are triangular → the mode is a real "always guess" floor)
        var counts = new Dictionary<int, int>(); foreach (var (_, t) in test) counts[t] = counts.GetValueOrDefault(t) + 1;
        var baseline = test.Count == 0 ? 0 : counts.Values.Max() / (double)test.Count;

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
            if (ep % 15 == 0 || ep == passes)
            {
                var tr = Eval(trainProbe);
                var te = Eval(test);
                if (te > bestTe) bestTe = te;
                if (tr > bestTr) bestTr = tr;
                curve.Add((int)Math.Round(te * 100));
                if (tr >= 0.99 && te >= 0.97) break;   // solved — stop grinding
            }
        }
        return (bestTr, bestTe, baseline, string.Join("→", curve));
    }
}
