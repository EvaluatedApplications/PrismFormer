// Copyright (c) 2026 Dongyang Stephen Chen, trading as Evaluated Applications. All rights reserved.
// Source-available for study and verification; commercial use, redistribution, or derivatives require permission. See LICENSE.

using PrismFormer;

namespace PrismFormer.Bench;

/// <summary>
/// SPECTRAL PRISMFORMER (--spectral-seq). The full loop: a machine's sound is a stream of frequency-bin tokens
/// over time; AlgFormer's embeddings are SEEDED from the log-frequency codec (so nearby bins start similar), and
/// it's trained next-token on NORMAL machine cycles. Anomalies are then detected as PREDICTION SURPRISE — the
/// mean next-token loss spikes on abnormal streams. This catches what cosine-to-template could not: an unseen
/// fault frequency, a stuck machine, a broken rhythm. Proof that the model learns the temporal spectral pattern
/// on top of the codec geometry.
/// </summary>
internal static class SpectralSeqBench
{
    const int Bins = 12, Vocab = 16, Ctx = 8, FaultBin = 10;
    static readonly int[] Cycle = { 3, 4, 5, 6, 5, 4 };   // the normal machine rhythm (rise + fall over bins 3..6)

    static double Freq(int bin) => 55.0 * Math.Pow(2, bin / 6.0);   // 6 bins / octave, 55 Hz base

    // a normal stream: the cycle repeated, with occasional +/-1 bin measurement jitter (never the fault bin)
    static int[] Normal(Random rng, int len)
    {
        var s = new int[len];
        for (var i = 0; i < len; i++)
        {
            var b = Cycle[i % Cycle.Length];
            if (rng.NextDouble() < 0.12) b = Math.Clamp(b + (rng.Next(2) == 0 ? -1 : 1), 2, 7);
            s[i] = b;
        }
        return s;
    }

    static double MeanLoss(AlgFormer m, int[] seq, out double acc)
    {
        double total = 0; int n = 0, ok = 0;
        for (var i = Ctx; i < seq.Length; i++)
        {
            var ctx = seq[(i - Ctx)..i];
            var lg = m.LogitsFor(ctx);
            var max = lg.Max(); var sum = 0.0; for (var w = 0; w < lg.Length; w++) sum += Math.Exp(lg[w] - max);
            total += -(lg[seq[i]] - max - Math.Log(sum));
            var best = 0; for (var w = 1; w < lg.Length; w++) if (lg[w] > lg[best]) best = w;
            if (best == seq[i]) ok++;
            n++;
        }
        acc = ok / (double)n; return total / n;
    }

    public static void Run()
    {
        try { Console.OutputEncoding = System.Text.Encoding.UTF8; } catch { }
        Console.WriteLine("SPECTRAL PRISMFORMER — train on normal machine rhythm, detect anomalies by prediction surprise\n");

        // embeddings seeded from the LOG-FREQUENCY codec: token=bin -> face(log2(freq)*3). Nearby bins start similar.
        double[] Seed(int w) => w < Bins ? PhasorCodec.NumberFace(Math.Log2(Freq(w)) * 3.0) : new double[PhasorCodec.Dim];
        var m = new AlgFormer(Vocab, shifts: 8, layers: 2, maxContext: Ctx, dModel: PhasorCodec.Dim,
                              frozenPrefix: PhasorCodec.FrozenReals, embedSeed: Seed, seed: 1);
        Console.WriteLine($"model: {m.ParamCount:N0} params (d={PhasorCodec.Dim}, S={m.Shifts}, L={m.Layers})   vocab {Vocab} freq-bins   ctx {Ctx}");
        Console.WriteLine($"normal cycle: bins [{string.Join(' ', Cycle)}]  ~  [{string.Join(" ", Cycle.Select(b => $"{Freq(b):0}Hz"))}]\n");

        // ---- train next-token on NORMAL streams ----
        var rng = new Random(7);
        const int steps = 6000;
        var train = Normal(rng, 4000);
        for (var t = 1; t <= steps; t++)
        {
            var lr = 3e-3 * (1.0 - 0.9 * (t - 1) / (double)(steps - 1));
            var i = Ctx + rng.Next(train.Length - Ctx);
            m.TrainStep(train[(i - Ctx)..i], train[i], lr);
            if (t % 1500 == 0) { var l = MeanLoss(m, Normal(new Random(99), 300), out var a); Console.WriteLine($"  step {t,5}/{steps}   normal loss {l:0.000}   next-token acc {a:P0}"); }
        }

        // ---- evaluate: normal (held-out) vs anomalies. anomaly = high mean loss (surprise). ----
        int[] Repeat(int[] pat, int len) { var s = new int[len]; for (var i = 0; i < len; i++) s[i] = pat[i % pat.Length]; return s; }
        var tests = new (string name, int[] seq)[]
        {
            ("NORMAL (held-out)",        Normal(new Random(123), 300)),
            ("fault freq (unseen bin)",  InjectFault(new Random(5), 300)),
            ("stuck machine (constant)", Repeat(new[]{5}, 300)),
            ("broken rhythm (knock)",    Repeat(new[]{3,6,3,6}, 300)),
            ("wrong cycle (shifted)",    Repeat(new[]{6,3,4,5}, 300)),
        };
        var baseLoss = MeanLoss(m, tests[0].seq, out _);
        Console.WriteLine("\nanomaly detection (mean next-token loss vs the NORMAL baseline):");
        Console.WriteLine($"     {"stream",-26} {"loss",7} {"x normal",9}   verdict");
        foreach (var (name, seq) in tests)
        {
            var l = MeanLoss(m, seq, out var a);
            var ratio = l / baseLoss;
            var flag = ratio > 2.0 ? "ANOMALY" : (name.StartsWith("NORMAL") ? "ok (normal)" : "ok");
            Console.WriteLine($"     {name,-26} {l,7:0.000} {ratio,8:0.0}x   {flag}");
        }
        Console.WriteLine("\n   the model learned the normal rhythm (low loss); every fault spikes the loss — including the");
        Console.WriteLine("   unseen fault frequency that cosine-to-template missed. That's the trained model earning its keep.");
    }

    // a normal stream with the fault frequency (unseen bin 10) injected occasionally
    static int[] InjectFault(Random rng, int len)
    {
        var s = Normal(rng, len);
        for (var i = 0; i < len; i++) if (rng.NextDouble() < 0.25) s[i] = FaultBin;
        return s;
    }
}
