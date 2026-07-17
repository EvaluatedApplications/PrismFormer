// Copyright (c) 2026 Dongyang Stephen Chen, trading as Evaluated Applications. All rights reserved.
// Source-available for study and verification; commercial use, redistribution, or derivatives require permission. See LICENSE.

using PrismFormer;

namespace PrismFormer.Bench;

/// <summary>
/// CAN YOU AVERAGE SEPARATELY-TRAINED MODELS? (--average). The standard NN result: averaging INDEPENDENTLY-initialised
/// nets gives garbage (loss-landscape permutation symmetry) — model soups / FedAvg only work because every member
/// shares ONE init or is re-synced each round (same basin). This tests whether PrismFormer's frozen codec changes
/// that. Train two models to competence on add(a,b), average their weights, measure the average — across INIT
/// (same seed + different data order = shared basin, vs different seed = different basin) x CODEC (frozen shared
/// number faces, vs none/random embeddings). If diff-init averaging is garbage WITHOUT the codec but competent WITH
/// it, the shared algebraic coordinate system is what lets genuinely-separate models average. If diff-init is garbage
/// either way, skill-by-averaging needs a shared basin here too (the codec pins only the input, not the computation).
/// </summary>
internal static class XferBench
{
    const int Hi = 12, V = 64, PLUS = 30, EQ = 31;

    static (int[] ctx, int tgt)[] Data()
    {
        var d = new List<(int[], int)>();
        for (var a = 0; a <= Hi; a++) for (var b = 0; b <= Hi; b++) d.Add((new[] { a, PLUS, b, EQ }, a + b));
        return d.ToArray();
    }
    static double[] Seed(int w) => w < 30 ? PhasorCodec.NumberFace(w) : PhasorCodec.Encode(w == PLUS ? "+" : "=");
    static AlgFormer Fresh(int initSeed, bool frozen) => new(V, shifts: 8, layers: 2, maxContext: 4, dModel: PhasorCodec.Dim,
        frozenPrefix: frozen ? PhasorCodec.FrozenReals : 0, embedSeed: frozen ? Seed : null, seed: initSeed);
    static void Train(AlgFormer m, (int[] ctx, int tgt)[] data, int dataSeed)
    {
        var rng = new Random(dataSeed); const int steps = 5000;
        for (var t = 1; t <= steps; t++) { var lr = 3e-3 * (1.0 - 0.9 * (t - 1) / (double)(steps - 1)); var (c, y) = data[rng.Next(data.Length)]; m.TrainStep(c, y, lr); }
    }
    static double Acc(AlgFormer m, (int[] ctx, int tgt)[] s) { var ok = 0; foreach (var (c, y) in s) if (m.Predict(c) == y) ok++; return ok / (double)s.Length; }

    static AlgFormer Average(AlgFormer a, AlgFormer b)   // element-wise mean of the two weight vectors (identical layout)
    {
        var x = a.Serialize(); var y = b.Serialize(); const int hdr = 24; var n = (x.Length - hdr) / 8;
        for (var i = 0; i < n; i++) BitConverter.GetBytes((BitConverter.ToDouble(x, hdr + i * 8) + BitConverter.ToDouble(y, hdr + i * 8)) / 2.0).CopyTo(x, hdr + i * 8);
        return AlgFormer.Deserialize(x);
    }

    public static void Run()
    {
        try { Console.OutputEncoding = System.Text.Encoding.UTF8; } catch { }
        var data = Data();
        Console.WriteLine("CAN YOU AVERAGE SEPARATELY-TRAINED MODELS? — train A + B on add(0..12), average the weights, test the average\n");
        Console.WriteLine($"  {"codec",-8} {"init",-24} {"A acc",7} {"B acc",7} {"AVERAGE acc",13}");
        foreach (var frozen in new[] { true, false })
        {
            foreach (var (label, seedA, seedB) in new[] { ("same init, diff data", 1, 1), ("different init", 1, 2) })
            {
                var A = Fresh(seedA, frozen); Train(A, data, 10);
                var B = Fresh(seedB, frozen); Train(B, data, 20);
                var C = Average(A, B);
                Console.WriteLine($"  {(frozen ? "frozen" : "none"),-8} {label,-24} {Acc(A, data),6:P0} {Acc(B, data),6:P0} {Acc(C, data),12:P0}");
            }
        }
        Console.WriteLine("\n  Same-init models share a basin -> the average stays competent (this is the model-soup / FedAvg regime).");
        Console.WriteLine("  Different-init models sit in different basins -> one-shot averaging collapses regardless of codec.\n");

        // CONTINUOUS coupling: does an elastic pull DURING training drag two DIFFERENT-init models into a shared basin
        // (so they converge + become averageable), where one-shot averaging failed? With vs without the codec.
        Console.WriteLine("  CONTINUOUS elastic coupling during training (pull to mean every 250 steps, alpha 0.15):");
        Console.WriteLine($"  {"codec",-8} {"init",-16} {"A acc",7} {"B acc",7} {"AVERAGE acc",13} {"divergence",11}");
        foreach (var frozen in new[] { true, false })
            foreach (var (label, sA, sB) in new[] { ("different init", 1, 2) })
            {
                var (a, b, avg, div) = Continuous(data, frozen, sA, sB);
                Console.WriteLine($"  {(frozen ? "frozen" : "none"),-8} {label,-16} {a,6:P0} {b,6:P0} {avg,12:P0} {div,11:0.000}");
            }
        Console.WriteLine("\n  If continuous coupling brings the different-init models together (divergence low + AVERAGE competent) where");
        Console.WriteLine("  one-shot averaging (9%) failed, then it's the CONTINUOUS pull — not the init — that makes models averageable;");
        Console.WriteLine("  the codec (frozen row) just makes it more robust. That is EASGD, kept in-basin so skill can flow.");
    }

    static double Div(AlgFormer a, AlgFormer b)
    { var x = a.Serialize(); var y = b.Serialize(); var n = (x.Length - 24) / 8; double s = 0; for (var i = 0; i < n; i++) { var d = BitConverter.ToDouble(x, 24 + i * 8) - BitConverter.ToDouble(y, 24 + i * 8); s += d * d; } return Math.Sqrt(s / n); }

    static (double a, double b, double avg, double div) Continuous((int[] ctx, int tgt)[] data, bool frozen, int seedA, int seedB)
    {
        var A = Fresh(seedA, frozen); var B = Fresh(seedB, frozen);
        var ra = new Random(10); var rb = new Random(20); const int steps = 5000; const double alpha = 0.15;
        for (var t = 1; t <= steps; t++)
        {
            var lr = 3e-3 * (1.0 - 0.9 * (t - 1) / (double)(steps - 1));
            var (ca, ya) = data[ra.Next(data.Length)]; A.TrainStep(ca, ya, lr);
            var (cb, yb) = data[rb.Next(data.Length)]; B.TrainStep(cb, yb, lr);
            if (t % 250 == 0)   // elastic pull: nudge both toward their mean
            {
                var xa = A.Serialize(); var xb = B.Serialize(); const int hdr = 24; var n = (xa.Length - hdr) / 8;
                for (var i = 0; i < n; i++)
                {
                    var va = BitConverter.ToDouble(xa, hdr + i * 8); var vb = BitConverter.ToDouble(xb, hdr + i * 8); var m = (va + vb) / 2;
                    BitConverter.GetBytes((1 - alpha) * va + alpha * m).CopyTo(xa, hdr + i * 8);
                    BitConverter.GetBytes((1 - alpha) * vb + alpha * m).CopyTo(xb, hdr + i * 8);
                }
                A = AlgFormer.Deserialize(xa); B = AlgFormer.Deserialize(xb);
            }
        }
        return (Acc(A, data), Acc(B, data), Acc(Average(A, B), data), Div(A, B));
    }
}
