// Copyright (c) 2026 Dongyang Stephen Chen, trading as Evaluated Applications. All rights reserved.
// Source-available for study and verification; commercial use, redistribution, or derivatives require permission. See LICENSE.

using PrismFormer;

namespace PrismFormer.Bench;

/// <summary>
/// BASAL COMPETENCY probe (--lesion), Levin-inspired. (A) LESION TOLERANCE: train the model, then kill a random
/// fraction of its weights at inference (via Serialize -> zero doubles -> Deserialize, no lib change) and watch
/// accuracy degrade — holographic/distributed representations should fail gracefully, not off a cliff. (B)
/// REGENERATION: take a damaged model and let it retrain — does it heal back to the goal, and faster than a fresh
/// model reaches it from scratch (surviving structure accelerates recovery)? Basal competency + anatomical
/// homeostasis for a neural net.
/// </summary>
internal static class LesionBench
{
    const int Hi = 12, V = 64, PLUS = 30, EQ = 31;   // tokens: 0..29 numbers, 30 '+', 31 '='

    static (int[] ctx, int tgt)[] Data()
    {
        var d = new List<(int[], int)>();
        for (var a = 0; a <= Hi; a++) for (var b = 0; b <= Hi; b++) d.Add((new[] { a, PLUS, b, EQ }, a + b));
        return d.ToArray();
    }
    static double[] Seed(int w) => w < 30 ? PhasorCodec.NumberFace(w) : PhasorCodec.Encode(w == PLUS ? "+" : "=");

    static AlgFormer Fresh(int seed) => new(V, shifts: 8, layers: 2, maxContext: 4, dModel: PhasorCodec.Dim, frozenPrefix: PhasorCodec.FrozenReals, embedSeed: Seed, seed: seed);
    static void Train(AlgFormer m, (int[] ctx, int tgt)[] data, int steps, int seed)
    {
        var rng = new Random(seed);
        for (var t = 1; t <= steps; t++) { var lr = 3e-3 * (1.0 - 0.9 * (t - 1) / (double)(steps - 1)); var (c, y) = data[rng.Next(data.Length)]; m.TrainStep(c, y, lr); }
    }
    static double Acc(AlgFormer m, (int[] ctx, int tgt)[] data) { var ok = 0; foreach (var (c, y) in data) if (m.Predict(c) == y) ok++; return ok / (double)data.Length; }

    // damage: zero a random fraction of the model's weight doubles (Serialize header = 6 ints = 24 bytes)
    static AlgFormer Lesion(AlgFormer m, double frac, int seed)
    {
        var b = m.Serialize(); const int hdr = 24; var nD = (b.Length - hdr) / 8; var rng = new Random(seed);
        for (var i = 0; i < nD; i++) if (rng.NextDouble() < frac) for (var j = 0; j < 8; j++) b[hdr + i * 8 + j] = 0;
        return AlgFormer.Deserialize(b);
    }

    public static void Run()
    {
        try { Console.OutputEncoding = System.Text.Encoding.UTF8; } catch { }
        var data = Data();
        Console.WriteLine("BASAL COMPETENCY — lesion tolerance + regeneration (add a+b, operands 0..12)\n");

        var m = Fresh(1); Train(m, data, 5000, 1);
        var baseAcc = Acc(m, data);
        Console.WriteLine($"  trained baseline accuracy: {baseAcc:P0}\n");

        // (A) lesion tolerance: zero an increasing fraction of weights, measure accuracy (mean over 5 random masks)
        Console.WriteLine("  (A) LESION TOLERANCE — kill a random fraction of weights at inference:");
        Console.WriteLine($"      {"weights killed",15} {"accuracy",9}");
        foreach (var f in new[] { 0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70 })
        {
            double acc = 0; var reps = f == 0 ? 1 : 5;
            for (var r = 0; r < reps; r++) acc += Acc(Lesion(m, f, 100 + r), data);
            Console.WriteLine($"      {f,14:P0} {acc / reps,9:P0}");
        }

        // (B) regeneration: damage hard, then retrain; compare recovery vs a fresh model over the same steps
        Console.WriteLine("\n  (B) REGENERATION — damage 40% of weights, then retrain (vs a fresh model from scratch):");
        var hurt = Lesion(m, 0.40, 7);
        Console.WriteLine($"      after 40% lesion:            {Acc(hurt, data):P0}");
        var scratch = Fresh(9);
        foreach (var s in new[] { 500, 1500, 3000 })
        {
            Train(hurt, data, s, 200 + s);      // continue healing the damaged model
            Train(scratch, data, s, 300 + s);   // fresh model gets the same extra steps
            Console.WriteLine($"      +{s,4} steps:  regenerated {Acc(hurt, data),5:P0}   |   from-scratch {Acc(scratch, data),5:P0}");
        }
        Console.WriteLine("\n  graceful degradation = distributed/holographic competency; regeneration outpacing scratch = the surviving");
        Console.WriteLine("  structure carries the goal and heals toward it — basal competency + homeostasis, not coded in explicitly.");
    }
}
