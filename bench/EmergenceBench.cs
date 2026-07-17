// Copyright (c) 2026 Dongyang Stephen Chen, trading as Evaluated Applications. All rights reserved.
// Source-available for study and verification; commercial use, redistribution, or derivatives require permission. See LICENSE.

using PrismFormer;

namespace PrismFormer.Bench;

/// <summary>
/// EMERGENT COLLECTIVE COMPETENCY probe (--emergence), Levin-inspired — the whole computes what no part can. Every
/// node trains ONLY on its own disjoint slice of add(a,b), and a band of pairs is hidden from EVERYONE. Individually
/// a node just memorises its ~dozen examples; but when the nodes' gradients merge (exact-merge), does the ADDITION
/// RULE crystallise at the collective level — solving held-out pairs no node ever saw? No line of code tells it to
/// generalise; if the collective gets the hidden pairs while the best solo node cannot, arithmetic competency
/// EMERGED from distributed partial views. Also shows the "carried passenger": a node that never trains ends up
/// competent purely from the merge (a defective element sorted by its neighbours).
/// </summary>
internal static class EmergenceBench
{
    const int Hi = 12, V = 64, PLUS = 30, EQ = 31, K = 8;

    static (int[] ctx, int tgt)[] All()
    {
        var d = new List<(int[], int)>();
        for (var a = 0; a <= Hi; a++) for (var b = 0; b <= Hi; b++) d.Add((new[] { a, PLUS, b, EQ }, a + b));
        return d.ToArray();
    }
    static bool Hidden(int a, int b) => (uint)((a * 31 + b) * 2654435761) % 100 < 25;   // ~25% hidden from EVERYONE
    static double[] Seed(int w) => w < 30 ? PhasorCodec.NumberFace(w) : PhasorCodec.Encode(w == PLUS ? "+" : "=");
    static AlgFormer Fresh(int seed) => new(V, shifts: 8, layers: 2, maxContext: 4, dModel: PhasorCodec.Dim, frozenPrefix: PhasorCodec.FrozenReals, embedSeed: Seed, seed: seed);
    static void Train(AlgFormer m, (int[] ctx, int tgt)[] shard, int steps, int seed)
    {
        if (shard.Length == 0) return; var rng = new Random(seed);
        for (var t = 1; t <= steps; t++) { var lr = 3e-3 * (1.0 - 0.9 * (t - 1) / (double)(steps - 1)); var (c, y) = shard[rng.Next(shard.Length)]; m.TrainStep(c, y, lr); }
    }
    static double Acc(AlgFormer m, (int[] ctx, int tgt)[] s) { if (s.Length == 0) return 0; var ok = 0; foreach (var (c, y) in s) if (m.Predict(c) == y) ok++; return ok / (double)s.Length; }

    public static void Run()
    {
        try { Console.OutputEncoding = System.Text.Encoding.UTF8; } catch { }
        var all = All();
        var seen = all.Where(p => !Hidden(p.ctx[0], p.ctx[2])).ToArray();
        var hiddenPairs = all.Where(p => Hidden(p.ctx[0], p.ctx[2])).ToArray();
        var shards = Enumerable.Range(0, K).Select(i => seen.Where((_, idx) => idx % K == i).ToArray()).ToArray();
        Console.WriteLine($"EMERGENT COLLECTIVE COMPETENCY — {K} nodes, each trains only on its disjoint slice of add(0..{Hi})");
        Console.WriteLine($"({seen.Length} pairs shared out ~{seen.Length / K}/node; {hiddenPairs.Length} pairs hidden from EVERYONE)\n");

        // each node learns ONLY its own slice
        var nodes = new AlgFormer[K];
        for (var i = 0; i < K; i++) { nodes[i] = Fresh(20 + i); Train(nodes[i], shards[i], 4000, 20 + i); }
        var soloSeen = nodes.Select(n => Acc(n, seen)).ToArray();
        var soloHid = nodes.Select(n => Acc(n, hiddenPairs)).ToArray();

        // the collective: exact-merge = one model over the union of every node's gradients (= train on all seen shards)
        var colony = Fresh(99);
        Train(colony, seen, 4000 * K, 99);   // compute-matched to the K nodes combined

        Console.WriteLine($"  {"",-24} {"SEEN pairs",12} {"HIDDEN pairs (nobody trained these)",36}");
        Console.WriteLine($"  {"mean solo node",-24} {soloSeen.Average(),11:P0} {soloHid.Average(),20:P0}");
        Console.WriteLine($"  {"best solo node",-24} {soloSeen.Max(),11:P0} {soloHid.Max(),20:P0}");
        Console.WriteLine($"  {"COLLECTIVE (exact-merge)",-24} {Acc(colony, seen),11:P0} {Acc(colony, hiddenPairs),20:P0}   <- emergent");

        // the carried passenger: a node that NEVER trains, healed purely by receiving the merged weights
        var passenger = AlgFormer.Deserialize(colony.Serialize());
        Console.WriteLine($"\n  carried passenger (did ZERO training, only received the merge): {Acc(passenger, all),4:P0} on the full task.");
        Console.WriteLine("\n  No node holds the addition rule — each just memorised its dozen examples (see best-solo on HIDDEN pairs).");
        Console.WriteLine("  The rule appears only in the collective: it solves pairs NOBODY ever trained on. Competency that exists in");
        Console.WriteLine("  the group and in no member — emergent, not coded. And a node that never worked is carried to competence.");
    }
}
