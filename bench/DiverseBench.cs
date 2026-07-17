// Copyright (c) 2026 Dongyang Stephen Chen, trading as Evaluated Applications. All rights reserved.
// Source-available for study and verification; commercial use, redistribution, or derivatives require permission. See LICENSE.

using PrismFormer;

namespace PrismFormer.Bench;

/// <summary>
/// HETEROGENEOUS COLLECTIVE probe (--diverse), Levin-inspired — the "multiple sorting algorithms grouped together"
/// analog. A population of DIFFERENT-config models (varied shift-count = receptive field, varied depth) cannot merge
/// weights, so they act as an ensemble: each learns the task from its own vantage, then votes (averaged logits). On
/// an UNDERTRAINED task (single members ~partial) diverse members make DECORRELATED errors, so the diverse vote
/// should beat both any single member and a HOMOGENEOUS ensemble (same config, seed-only diversity). Tests whether
/// architectural diversity is a real collective advantage or just more of the same.
/// </summary>
internal static class DiverseBench
{
    const int Hi = 20, V = 64, PLUS = 41, EQ = 42, K = 6, STEPS = 1500;

    static (int[] ctx, int tgt)[] Data()
    {
        var d = new List<(int[], int)>();
        for (var a = 0; a <= Hi; a++) for (var b = 0; b <= Hi; b++) d.Add((new[] { a, PLUS, b, EQ }, a + b));
        return d.ToArray();
    }
    static double[] Seed(int w) => w < 41 ? PhasorCodec.NumberFace(w) : PhasorCodec.Encode(w == PLUS ? "+" : "=");
    static AlgFormer Model(int shifts, int layers, int seed) => new(V, shifts, layers, maxContext: 4, dModel: PhasorCodec.Dim, frozenPrefix: PhasorCodec.FrozenReals, embedSeed: Seed, seed: seed);
    static void Train(AlgFormer m, (int[] ctx, int tgt)[] data, int seed)
    {
        var rng = new Random(seed);
        for (var t = 1; t <= STEPS; t++) { var lr = 3e-3 * (1.0 - 0.9 * (t - 1) / (double)(STEPS - 1)); var (c, y) = data[rng.Next(data.Length)]; m.TrainStep(c, y, lr); }
    }
    static double Acc(AlgFormer m, (int[] ctx, int tgt)[] data) { var ok = 0; foreach (var (c, y) in data) if (m.Predict(c) == y) ok++; return ok / (double)data.Length; }

    static double Ensemble(AlgFormer[] ms, (int[] ctx, int tgt)[] data)
    {
        var ok = 0;
        foreach (var (c, y) in data)
        {
            var agg = new double[V];
            foreach (var m in ms) { var lg = m.LogitsFor(c); for (var w = 0; w < V; w++) agg[w] += lg[w]; }   // vote = mean logits
            var best = 0; for (var w = 1; w < V; w++) if (agg[w] > agg[best]) best = w;
            if (best == y) ok++;
        }
        return ok / (double)data.Length;
    }

    static (double mean, double best, double ens) Population(AlgFormer[] ms, (int[] ctx, int tgt)[] data)
    {
        var accs = ms.Select(m => Acc(m, data)).ToArray();
        return (accs.Average(), accs.Max(), Ensemble(ms, data));
    }

    static AlgFormer[] Build(int[] shifts, int[] layers, int seedBase, (int[] ctx, int tgt)[] data)
    {
        var pop = new AlgFormer[K];
        for (var i = 0; i < K; i++) { pop[i] = Model(shifts[i], layers[i], seedBase + i); Train(pop[i], data, seedBase + i); }
        return pop;
    }

    public static void Run()
    {
        try { Console.OutputEncoding = System.Text.Encoding.UTF8; } catch { }
        var data = Data();
        Console.WriteLine($"HETEROGENEOUS COLLECTIVE — {K} models vote on add(a,b), operands 0..{Hi}, undertrained ({STEPS} steps each)\n");

        var pops = new (string name, int[] shifts, int[] layers, int seed)[]
        {
            ("homogeneous",       new[]{ 8, 8, 8, 8, 8, 8 },      new[]{ 2,2,2,2,2,2 }, 50),   // seed-diversity only
            ("diverse (matched)", new[]{ 8,10,12,14,16,20 },      new[]{ 2,2,2,2,2,2 }, 70),   // different receptive fields, all capable
            ("diverse (w/ weak)", new[]{ 4, 6, 8,12,16,24 },      new[]{ 1,2,2,1,2,2 }, 10),   // includes underpowered configs
        };
        Console.WriteLine($"  {"population",-18} {"mean member",12} {"best member",12} {"ENSEMBLE vote",14} {"vs best",8}");
        foreach (var (name, sh, ly, sd) in pops)
        {
            var (m, b, e) = Population(Build(sh, ly, sd, data), data);
            Console.WriteLine($"  {name,-18} {m,11:P0} {b,11:P0} {e,13:P0} {(e - b >= 0 ? "+" : "") + (e - b).ToString("P0"),8}");
        }
        Console.WriteLine("\n  The vote beats the best single member when members err on DIFFERENT items (decorrelated errors). Naive");
        Console.WriteLine("  architectural diversity that mixes in underpowered configs HURTS — it pollutes the vote with bad members.");
        Console.WriteLine("  'Multiple algorithms grouped' only wins when each is individually competent AND they fail differently.");
    }
}
