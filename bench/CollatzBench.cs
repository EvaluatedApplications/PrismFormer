// Copyright (c) 2026 Dongyang Stephen Chen, trading as Evaluated Applications. All rights reserved.
// Source-available for study and verification; commercial use, redistribution, or derivatives require permission. See LICENSE.

using PrismFormer;

namespace PrismFormer.Bench;

/// <summary>
/// COLLATZ stopping-time probe (--collatz). NOT a proof — a map of how much of Collatz is LEARNABLE. Given n (as its
/// phasor number face), predict its total stopping time s(n) = steps of (n→3n+1 if odd, n/2 if even) to reach 1.
/// Trained on n in [1,HI], tested on HELD-OUT n and EXTRAPOLATED to [HI+1,2·HI]. The number codec makes nearby integers
/// quasi-orthogonal (its linear band) yet shares a continuous magnitude signal (its log band), so the honest prediction
/// is: the model captures the TREND (s grows ~with log n, carried by the log band) but misses the erratic per-n spikes
/// (they live in the orthogonal band — essentially unpredictable without running the trajectory, like the hash wall).
/// We report exact / within-tolerance / correlation to separate "learned the shape" from "learned the spikes".
/// </summary>
internal static class CollatzBench
{
    const int HI = 2048, MAXN = 4096, EQ = 4097, V = 4100;   // number tokens 0..4096 (covers n and s), EQ=4097

    static int Stop(long n) { var s = 0; while (n > 1) { n = (n & 1) == 0 ? n / 2 : 3 * n + 1; s++; if (s > 100000) return s; } return s; }
    static double[] Seed(int w) => w <= MAXN ? PhasorCodec.NumberFace(w) : PhasorCodec.Encode("=");

    static (double exact, double w5, double w10, double corr) Eval(AlgFormer m, IReadOnlyList<int> ns)
    {
        int ex = 0, a5 = 0, a10 = 0, n = ns.Count;
        var pr = new double[n]; var tr = new double[n];
        for (var i = 0; i < n; i++)
        {
            var t = Stop(ns[i]); var p = m.Predict(new[] { ns[i], EQ });
            pr[i] = p; tr[i] = t;
            if (p == t) ex++; if (Math.Abs(p - t) <= 5) a5++; if (Math.Abs(p - t) <= 10) a10++;
        }
        return (ex / (double)n, a5 / (double)n, a10 / (double)n, Pearson(pr, tr));
    }
    static double Pearson(double[] a, double[] b)
    {
        double ma = a.Average(), mb = b.Average(), sab = 0, saa = 0, sbb = 0;
        for (var i = 0; i < a.Length; i++) { var da = a[i] - ma; var db = b[i] - mb; sab += da * db; saa += da * da; sbb += db * db; }
        return saa > 0 && sbb > 0 ? sab / Math.Sqrt(saa * sbb) : 0;
    }

    public static void Run()
    {
        try { Console.OutputEncoding = System.Text.Encoding.UTF8; } catch { }
        Console.WriteLine($"COLLATZ stopping-time probe — predict s(n) from n's number face. train n∈[1,{HI}], test held-out + extrapolate to [{HI + 1},{2 * HI}].");
        Console.WriteLine("  NOT a proof — measures how much structure PrismFormer finds in a famously chaotic sequence.\n");

        var rng = new Random(7);
        var train = new List<(int[] ctx, int tgt)>(); var held = new List<int>();
        for (var nn = 1; nn <= HI; nn++) { if (rng.NextDouble() < 0.2) held.Add(nn); else train.Add((new[] { nn, EQ }, Stop(nn))); }
        var extrap = Enumerable.Range(HI + 1, HI).ToList();
        var maxTrainS = train.Max(x => x.tgt);
        Console.WriteLine($"  train {train.Count} / held-out {held.Count} / extrapolate {extrap.Count}.  (max stopping time in range ≈ {maxTrainS})\n");

        var m = new AlgFormer(V, shifts: 12, layers: 4, maxContext: 2, dModel: PhasorCodec.Dim, frozenPrefix: PhasorCodec.FrozenReals, embedSeed: Seed, seed: 1);
        Console.WriteLine($"  PrismFormer {m.ParamCount:N0} params\n");
        var data = train.Select(x => (x.ctx, x.tgt)).ToList();
        const int epochs = 300;
        for (var ep = 1; ep <= epochs; ep++)
        {
            var lr = 3e-3 * (1.0 - 0.9 * (ep - 1) / (double)(epochs - 1));
            m.TrainEpoch(data, 256, lr, shuffleSeed: ep);
            if (ep % 60 == 0 || ep == epochs)
            {
                var (tex, _, _, tcorr) = Eval(m, train.Select(x => x.ctx[0]).ToList());
                var (hex, _, _, hcorr) = Eval(m, held);
                Console.WriteLine($"    epoch {ep,3}: train exact {tex,5:P0} corr {tcorr:0.00}  |  held exact {hex,5:P0} corr {hcorr:0.00}");
            }
        }

        Console.WriteLine("\n  RESULT — exact match / within ±5 / within ±10 / correlation with true stopping time:");
        Console.WriteLine($"    {"set",-14} {"exact",7} {"±5",7} {"±10",7} {"corr",7}");
        foreach (var (name, ns) in new[] { ("train", train.Select(x => x.ctx[0]).ToList()), ("HELD-OUT", held), ("EXTRAPOLATE", extrap) })
        {
            var (ex, a5, a10, corr) = Eval(m, ns);
            Console.WriteLine($"    {name,-14} {ex,7:P0} {a5,7:P0} {a10,7:P0} {corr,7:0.00}");
        }
        Console.WriteLine("\n  High train / low held-out exact = memorised the spikes, can't predict unseen (the orthogonal band = chaos, like the");
        Console.WriteLine("  hash wall). Positive held-out CORRELATION with near-zero exact = it learned the TREND (magnitude→length via the log");
        Console.WriteLine("  band) but not the erratic per-n value. That gap is the honest answer: Collatz's shape is learnable, its spikes aren't.");
    }
}
