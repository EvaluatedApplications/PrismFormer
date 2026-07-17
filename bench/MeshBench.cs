// Copyright (c) 2026 Dongyang Stephen Chen, trading as Evaluated Applications. All rights reserved.
// Source-available for study and verification; commercial use, redistribution, or derivatives require permission. See LICENSE.

using PrismFormer;

namespace PrismFormer.Bench;

/// <summary>
/// FAITHFUL PRISM STUDIO MESH replica (--mesh). NOT the abandoned master/slave gradient-sum — this is the real colony:
/// N separate, autonomous models that "chatter". Every tick each node (a) trains its OWN head on its OWN local pairs
/// (no gradient sharing at all), then bleeds to a few peers exactly as SwarmChatter/StudioModel do: a PAIR-GOSSIP
/// (share one confident training example — the receiver adds it and learns it next epoch) and a WEIGHT-SLICE
/// ELASTIC-AVERAGE (a random 1024-double slice, receiver nudges mine = (1-a)*mine + a*theirs, a=0.05). Real knobs:
/// slice 1024, alpha 0.05, fanout 3. A PASSENGER node never trains — it only absorbs bleeds ("passive training",
/// HeadlessNode). Question: on the mechanism you actually run, does the collective still cohere and does the addition
/// RULE still emerge (hidden pairs nobody trained), with the nodes converging yet staying separate?
/// </summary>
internal static class MeshBench
{
    const int Hi = 12, V = 64, PLUS = 30, EQ = 31, N = 6;   // 6 worker nodes + 1 passenger
    const int SLICE = 1024, FANOUT = 3, TICKS = 200, STEPS = 300;
    const double ALPHA = 0.05;

    static (int[] ctx, int tgt)[] All()
    {
        var d = new List<(int[], int)>();
        for (var a = 0; a <= Hi; a++) for (var b = 0; b <= Hi; b++) d.Add((new[] { a, PLUS, b, EQ }, a + b));
        return d.ToArray();
    }
    static bool HiddenPair(int a, int b) => (uint)((a * 31 + b) * 2654435761) % 100 < 25;   // hidden from EVERYONE
    static double[] Seed(int w) => w < 30 ? PhasorCodec.NumberFace(w) : PhasorCodec.Encode(w == PLUS ? "+" : "=");
    static AlgFormer Fresh(int seed) => new(V, shifts: 8, layers: 2, maxContext: 4, dModel: PhasorCodec.Dim, frozenPrefix: PhasorCodec.FrozenReals, embedSeed: Seed, seed: seed);
    static void Train(AlgFormer m, List<(int[] ctx, int tgt)> pairs, int steps, Random rng)
    { if (pairs.Count == 0) return; for (var t = 0; t < steps; t++) { var (c, y) = pairs[rng.Next(pairs.Count)]; m.TrainStep(c, y, 2e-3); } }
    static double Acc(AlgFormer m, (int[] ctx, int tgt)[] s) { if (s.Length == 0) return 0; var ok = 0; foreach (var (c, y) in s) if (m.Predict(c) == y) ok++; return ok / (double)s.Length; }

    // ── chatter primitives (exactly StudioModel.WeightSlice / MergeWeightSlice, over the Serialize layout: 24-byte header) ──
    static (int start, double[] vals) WeightSlice(AlgFormer m, Random rng)
    {
        var b = m.Serialize(); var n = (b.Length - 24) / 8; var start = rng.Next(n - SLICE); var vals = new double[SLICE];
        for (var i = 0; i < SLICE; i++) vals[i] = BitConverter.ToDouble(b, 24 + (start + i) * 8);
        return (start, vals);
    }
    static AlgFormer MergeSlice(AlgFormer m, int start, double[] vals)   // mine = (1-a)*mine + a*theirs, on that slice only
    {
        var b = m.Serialize();
        for (var i = 0; i < vals.Length; i++) { var mine = BitConverter.ToDouble(b, 24 + (start + i) * 8); BitConverter.GetBytes((1 - ALPHA) * mine + ALPHA * vals[i]).CopyTo(b, 24 + (start + i) * 8); }
        return AlgFormer.Deserialize(b);   // frozen codec positions are identical across nodes, so blending them is a no-op
    }
    static double Divergence(AlgFormer a, AlgFormer b)
    { var x = a.Serialize(); var y = b.Serialize(); var n = (x.Length - 24) / 8; double s = 0; for (var i = 0; i < n; i++) { var d = BitConverter.ToDouble(x, 24 + i * 8) - BitConverter.ToDouble(y, 24 + i * 8); s += d * d; } return Math.Sqrt(s / n); }

    static int[] Peers(int self, int total, Random rng)   // FANOUT distinct peers != self
    {
        var pick = new List<int>(); while (pick.Count < FANOUT) { var p = rng.Next(total); if (p != self && !pick.Contains(p)) pick.Add(p); } return pick.ToArray();
    }

    public static void Run()
    {
        try { Console.OutputEncoding = System.Text.Encoding.UTF8; } catch { }
        var rng = new Random(7);
        var all = All();
        var seen = all.Where(p => !HiddenPair(p.ctx[0], p.ctx[2])).ToArray();
        var hidden = all.Where(p => HiddenPair(p.ctx[0], p.ctx[2])).ToArray();

        var total = N + 1; var PASS = N;   // node index N = the passenger
        var nodes = Enumerable.Range(0, total).Select(i => Fresh(20 + i)).ToArray();
        var local = Enumerable.Range(0, total).Select(_ => new List<(int[] ctx, int tgt)>()).ToArray();
        for (var i = 0; i < seen.Length; i++) local[i % N].Add(seen[i]);   // disjoint shards to the N workers; passenger gets NONE
        var seed0 = local[0].Count;

        Console.WriteLine($"FAITHFUL PRISM STUDIO MESH — {N} autonomous nodes chatter (weight-slice elastic avg a={ALPHA}, slice {SLICE}, fanout {FANOUT})");
        Console.WriteLine($"+ pair-gossip. NO gradient summing. {seen.Length} pairs disjointly sharded (~{seed0}/node); {hidden.Length} hidden from everyone; node {PASS} is a PASSENGER (never trains).\n");
        Console.WriteLine($"  {"tick",4} {"worker SEEN",12} {"worker HIDDEN",14} {"divergence",11} {"passenger(full)",16}");

        void Report(int tick)
        {
            var ws = Enumerable.Range(0, N).Average(i => Acc(nodes[i], seen));
            var wh = Enumerable.Range(0, N).Average(i => Acc(nodes[i], hidden));
            var div = Enumerable.Range(1, N - 1).Average(i => Divergence(nodes[0], nodes[i]));
            Console.WriteLine($"  {tick,4} {ws,11:P0} {wh,13:P0} {div,11:0.0000} {Acc(nodes[PASS], all),15:P0}");
        }
        Report(0);

        for (var tick = 1; tick <= TICKS; tick++)
        {
            for (var i = 0; i < N; i++) Train(nodes[i], local[i], STEPS, rng);   // each worker trains its OWN model on its OWN pairs

            // chatter: every node bleeds a pair + a weight slice to FANOUT peers (workers bleed; passenger only receives)
            for (var i = 0; i < N; i++)
            {
                var (start, vals) = WeightSlice(nodes[i], rng);
                foreach (var p in Peers(i, total, rng)) nodes[p] = MergeSlice(nodes[p], start, vals);
                if (local[i].Count > 0) { var pair = local[i][rng.Next(local[i].Count)]; var key = pair.ctx[0] * 100 + pair.ctx[2];
                    foreach (var p in Peers(i, N, rng)) if (!local[p].Any(q => q.ctx[0] * 100 + q.ctx[2] == key)) local[p].Add(pair); }
            }
            if (tick % 20 == 0 || tick == TICKS) Report(tick);
        }

        Console.WriteLine("\n  Read the table, not a canned conclusion. This is the mechanism you actually run (no gradient sum). Watch whether");
        Console.WriteLine("  the workers cohere in FUNCTION (seen% up) vs in WEIGHTS (divergence down or up), whether the addition rule");
        Console.WriteLine("  EMERGES on hidden pairs nobody trained, and whether the passive PASSENGER is ever carried by weight-slices alone.");
        Console.WriteLine("  If seen% rises while divergence also rises, consensus is FUNCTIONAL (pair-gossip shares data → each node learns the");
        Console.WriteLine("  rule in its own weight configuration — degeneracy), not STRUCTURAL (weight-bleed at a=0.05 is a gentle tether, not a merge).");
    }
}
