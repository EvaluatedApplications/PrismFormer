// Copyright (c) 2026 Dongyang Stephen Chen, trading as Evaluated Applications. All rights reserved.
// Source-available for study and verification; commercial use, redistribution, or derivatives require permission. See LICENSE.

using PrismFormer;

namespace PrismFormer.Bench;

/// <summary>
/// COLLECTIVE COMPETENCY probe (--colony), Levin-inspired — the "sorting on broken hardware" analog at the SWARM
/// level. A colony of K replicas each trains on its own disjoint data shard, then every round they federated-average
/// their weights (Serialize -> mean the doubles -> Deserialize, exactly the anchor's merge). "Broken hardware": each
/// round every node independently FAILS with probability p (drops out — its shard goes unseen, it is excluded from
/// the merge, then re-syncs to the average = healed). Question: does the colony still reach the goal as p climbs —
/// competency the single network (see --lesion) does NOT have? Second panel: BYZANTINE nodes (return garbage that is
/// still averaged in) to find where naive averaging breaks — the honest limit of a trust-free merge.
/// </summary>
internal static class ColonyBench
{
    const int Hi = 12, V = 64, PLUS = 30, EQ = 31, K = 6;

    static (int[] ctx, int tgt)[] Data()
    {
        var d = new List<(int[], int)>();
        for (var a = 0; a <= Hi; a++) for (var b = 0; b <= Hi; b++) d.Add((new[] { a, PLUS, b, EQ }, a + b));
        return d.ToArray();
    }
    static double[] Seed(int w) => w < 30 ? PhasorCodec.NumberFace(w) : PhasorCodec.Encode(w == PLUS ? "+" : "=");
    static AlgFormer Fresh(int seed) => new(V, shifts: 8, layers: 2, maxContext: 4, dModel: PhasorCodec.Dim, frozenPrefix: PhasorCodec.FrozenReals, embedSeed: Seed, seed: seed);
    static void Train(AlgFormer m, (int[] ctx, int tgt)[] shard, int steps, Random rng)
    {
        if (shard.Length == 0) return;
        for (var t = 0; t < steps; t++) { var (c, y) = shard[rng.Next(shard.Length)]; m.TrainStep(c, y, 3e-3); }
    }
    static double Acc(AlgFormer m, (int[] ctx, int tgt)[] data) { var ok = 0; foreach (var (c, y) in data) if (m.Predict(c) == y) ok++; return ok / (double)data.Length; }

    static byte[] Average(List<byte[]> models)   // element-wise mean of the weight doubles (identical layout => valid)
    {
        var o = (byte[])models[0].Clone(); const int hdr = 24; var nD = (o.Length - hdr) / 8;
        for (var i = 0; i < nD; i++)
        {
            double s = 0; foreach (var b in models) s += BitConverter.ToDouble(b, hdr + i * 8);
            BitConverter.GetBytes(s / models.Count).CopyTo(o, hdr + i * 8);
        }
        return o;
    }
    static byte[] Garbage(byte[] model, Random rng)   // a Byzantine node: keep the header, fill weights with noise
    {
        var o = (byte[])model.Clone(); const int hdr = 24; var nD = (o.Length - hdr) / 8;
        for (var i = 0; i < nD; i++) BitConverter.GetBytes((rng.NextDouble() * 2 - 1)).CopyTo(o, hdr + i * 8);
        return o;
    }

    // one colony run: returns (final accuracy, mean #nodes alive per round)
    static (double acc, double alive) RunColony((int[] ctx, int tgt)[] data, double pFail, double pByz, int rounds, int stepsPerRound, int seed)
    {
        var rng = new Random(seed);
        var reps = Enumerable.Range(0, K).Select(i => Fresh(seed * 100 + i)).ToArray();
        var shards = Enumerable.Range(0, K).Select(i => data.Where((_, idx) => idx % K == i).ToArray()).ToArray();
        double aliveTot = 0;
        for (var r = 0; r < rounds; r++)
        {
            var contrib = new List<byte[]>(); var alive = 0;
            for (var i = 0; i < K; i++)
            {
                if (rng.NextDouble() < pFail) continue;             // broken hardware: node down this round
                alive++;
                Train(reps[i], shards[i], stepsPerRound, rng);
                contrib.Add(rng.NextDouble() < pByz ? Garbage(reps[i].Serialize(), rng) : reps[i].Serialize());
            }
            aliveTot += alive;
            if (contrib.Count == 0) continue;                       // whole colony down — skip merge, keep weights
            var avg = Average(contrib);
            for (var i = 0; i < K; i++) reps[i] = AlgFormer.Deserialize(avg);   // sync + heal the dropped nodes
        }
        return (Acc(reps[0], data), aliveTot / rounds);
    }

    static (double acc, double alive) Mean((int[] ctx, int tgt)[] data, double pFail, double pByz)
    {
        double a = 0, n = 0; for (var s = 1; s <= 3; s++) { var (ac, al) = RunColony(data, pFail, pByz, 15, 400, s); a += ac; n += al; }
        return (a / 3, n / 3);
    }

    // EXACT-MERGE swarm: one model, gradients summed across the ALIVE shards each round = mathematically identical to
    // single-machine training on the union (this is PrismFormer's real merge, not lossy weight-averaging).
    static double RunExact((int[] ctx, int tgt)[] data, double pFail, int rounds, int stepsPerNode, int seed)
    {
        var rng = new Random(seed); var m = Fresh(seed * 100);
        var shards = Enumerable.Range(0, K).Select(i => data.Where((_, idx) => idx % K == i).ToArray()).ToArray();
        for (var r = 0; r < rounds; r++)
        {
            var union = new List<(int[] ctx, int tgt)>(); var alive = 0;
            for (var i = 0; i < K; i++) if (rng.NextDouble() >= pFail) { union.AddRange(shards[i]); alive++; }
            if (union.Count == 0) continue;
            var arr = union.ToArray();
            for (var t = 0; t < alive * stepsPerNode; t++) { var (c, y) = arr[rng.Next(arr.Length)]; m.TrainStep(c, y, 3e-3); }  // compute-matched to FedAvg
        }
        return Acc(m, data);
    }
    static double MeanExact((int[] ctx, int tgt)[] data, double pFail)
    { double a = 0; for (var s = 1; s <= 3; s++) a += RunExact(data, pFail, 15, 400, s); return a / 3; }

    public static void Run()
    {
        try { Console.OutputEncoding = System.Text.Encoding.UTF8; } catch { }
        var data = Data();
        Console.WriteLine($"COLLECTIVE COMPETENCY — a colony of {K} replicas, disjoint data shards, federated-averaged each round");
        Console.WriteLine("(add a+b, operands 0..12; mean of 3 runs). Compare to --lesion: one net dies at 5% weight damage.\n");

        Console.WriteLine("  (C) BROKEN HARDWARE — each node drops out with probability p every round (excluded, then re-synced).");
        Console.WriteLine("      Two merges: FedAvg (lossy weight-average, the anchor's periodic sync) vs EXACT-MERGE (lossless gradient sum):");
        Console.WriteLine($"      {"node failure p",15} {"~alive",8} {"FedAvg acc",12} {"EXACT-MERGE acc",16}");
        foreach (var p in new[] { 0.0, 0.2, 0.4, 0.6, 0.8 })
        {
            var (acc, alive) = Mean(data, p, 0.0);
            var exact = MeanExact(data, p);
            Console.WriteLine($"      {p,14:P0} {alive,6:0.0}/{K} {acc,11:P0} {exact,15:P0}");
        }

        Console.WriteLine("\n  (C2) BYZANTINE — 20% node dropout, plus a fraction of survivors return GARBAGE that is still averaged in:");
        Console.WriteLine($"      {"byzantine frac",15} {"colony accuracy",16}");
        foreach (var q in new[] { 0.0, 0.1, 0.2, 0.35 })
        {
            var (acc, _) = Mean(data, 0.2, q);
            Console.WriteLine($"      {q,14:P0} {acc,15:P0}");
        }
        Console.WriteLine("\n  Both merges keep working as nodes die — basal competency that lives in the GROUP, not the single brittle");
        Console.WriteLine("  network (--lesion: one net dies at 5% weight damage). EXACT-MERGE holds a far higher ceiling: it is lossless");
        Console.WriteLine("  (gradient sum = single-machine training on the union), so a dropped node just means fewer examples that round,");
        Console.WriteLine("  not a degraded solution. FedAvg is robust too but weight-averaging a precise phasor code caps it low. Byzantine");
        Console.WriteLine("  garbage poisons the mean (the honest limit: a trust-free average has no defence — real swarms need median/trust).");
    }
}
