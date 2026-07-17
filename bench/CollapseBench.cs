// Copyright (c) 2026 Dongyang Stephen Chen, trading as Evaluated Applications. All rights reserved.
// Source-available for study and verification; commercial use, redistribution, or derivatives require permission. See LICENSE.

using PrismFormer;

namespace PrismFormer.Bench;

/// <summary>
/// DOES THE BLEED DAMAGE THE HOLOGRAPHIC INFO? (--collapse). Tests the hypothesis: if elastic weight-averaging
/// corrupted the precise arithmetic code, algebra accuracy would periodically COLLAPSE as nodes absorb bleeds. Runs
/// the faithful mesh twice, identical but for one thing: FROZEN codec (the real Studio setup — number/identity faces
/// pinned, so bleeding them is a no-op) vs UNFROZEN codec (faces learnable → they diverge across nodes → a bleed
/// genuinely averages DIFFERENT number representations together, corrupting the decode). Measures seen-accuracy EVERY
/// tick and counts collapses (single-tick drops > 10 points). Prediction: frozen = smooth (matches the user's live
/// observation of no collapse); unfrozen = periodic collapses. That gap = the codec-pinning is what protects the
/// holographic information from the averaging.
/// </summary>
internal static class CollapseBench
{
    const int Hi = 12, V = 64, PLUS = 30, EQ = 31, N = 5;
    const int SLICE = 1024, FANOUT = 3, TICKS = 40, STEPS = 300;
    const double ALPHA = 0.05;

    static (int[] ctx, int tgt)[] All()
    {
        var d = new List<(int[], int)>();
        for (var a = 0; a <= Hi; a++) for (var b = 0; b <= Hi; b++) d.Add((new[] { a, PLUS, b, EQ }, a + b));
        return d.ToArray();
    }
    static double[] Seed(int w) => w < 30 ? PhasorCodec.NumberFace(w) : PhasorCodec.Encode(w == PLUS ? "+" : "=");
    static AlgFormer Fresh(int seed, bool frozen) => new(V, shifts: 8, layers: 2, maxContext: 4, dModel: PhasorCodec.Dim, frozenPrefix: frozen ? PhasorCodec.FrozenReals : 0, embedSeed: Seed, seed: seed);
    static void Train(AlgFormer m, List<(int[] ctx, int tgt)> pairs, int steps, Random rng)
    { if (pairs.Count == 0) return; for (var t = 0; t < steps; t++) { var (c, y) = pairs[rng.Next(pairs.Count)]; m.TrainStep(c, y, 2e-3); } }
    static double Acc(AlgFormer m, (int[] ctx, int tgt)[] s) { var ok = 0; foreach (var (c, y) in s) if (m.Predict(c) == y) ok++; return ok / (double)s.Length; }

    static (int start, double[] vals) WeightSlice(AlgFormer m, Random rng)
    { var b = m.Serialize(); var n = (b.Length - 24) / 8; var start = rng.Next(n - SLICE); var vals = new double[SLICE]; for (var i = 0; i < SLICE; i++) vals[i] = BitConverter.ToDouble(b, 24 + (start + i) * 8); return (start, vals); }
    static AlgFormer MergeSlice(AlgFormer m, int start, double[] vals)
    { var b = m.Serialize(); for (var i = 0; i < vals.Length; i++) { var mine = BitConverter.ToDouble(b, 24 + (start + i) * 8); BitConverter.GetBytes((1 - ALPHA) * mine + ALPHA * vals[i]).CopyTo(b, 24 + (start + i) * 8); } return AlgFormer.Deserialize(b); }
    static int[] Peers(int self, int total, Random rng) { var pick = new List<int>(); while (pick.Count < FANOUT) { var p = rng.Next(total); if (p != self && !pick.Contains(p)) pick.Add(p); } return pick.ToArray(); }

    static double[] RunMesh(bool frozen, (int[] ctx, int tgt)[] seen)
    {
        var rng = new Random(7);
        var nodes = Enumerable.Range(0, N).Select(i => Fresh(20 + i, frozen)).ToArray();
        var local = Enumerable.Range(0, N).Select(_ => new List<(int[] ctx, int tgt)>()).ToArray();
        for (var i = 0; i < seen.Length; i++) local[i % N].Add(seen[i]);
        var series = new double[TICKS + 1];
        series[0] = Enumerable.Range(0, N).Average(i => Acc(nodes[i], seen));
        for (var tick = 1; tick <= TICKS; tick++)
        {
            for (var i = 0; i < N; i++) Train(nodes[i], local[i], STEPS, rng);
            for (var i = 0; i < N; i++)
            {
                var (start, vals) = WeightSlice(nodes[i], rng);
                foreach (var p in Peers(i, N, rng)) nodes[p] = MergeSlice(nodes[p], start, vals);
                var pair = local[i][rng.Next(local[i].Count)]; var key = pair.ctx[0] * 100 + pair.ctx[2];
                foreach (var p in Peers(i, N, rng)) if (!local[p].Any(q => q.ctx[0] * 100 + q.ctx[2] == key)) local[p].Add(pair);
            }
            series[tick] = Enumerable.Range(0, N).Average(i => Acc(nodes[i], seen));
        }
        return series;
    }

    static (double maxDrop, int collapses) Stats(double[] s)
    { double maxDrop = 0; int col = 0; for (var t = 1; t < s.Length; t++) { var d = s[t - 1] - s[t]; if (d > maxDrop) maxDrop = d; if (d > 0.10) col++; } return (maxDrop, col); }

    public static void Run()
    {
        try { Console.OutputEncoding = System.Text.Encoding.UTF8; } catch { }
        var seen = All();
        Console.WriteLine("DOES THE BLEED DAMAGE THE HOLOGRAPHIC INFO? — faithful mesh, FROZEN vs UNFROZEN codec, algebra accuracy every tick\n");
        var fr = RunMesh(true, seen);
        var un = RunMesh(false, seen);
        Console.WriteLine($"  {"tick",4} {"FROZEN codec",13} {"UNFROZEN codec",15}   (watch for single-tick collapses)");
        for (var t = 0; t <= TICKS; t += 2)
            Console.WriteLine($"  {t,4} {fr[t],12:P0} {un[t],14:P0}   {(t > 0 && fr[t - 1] - fr[t] > 0.10 ? "F-COLLAPSE " : "")}{(t > 0 && un[t - 1] - un[t] > 0.10 ? "U-COLLAPSE" : "")}");
        var (fd, fc) = Stats(fr); var (ud, uc) = Stats(un);
        Console.WriteLine($"\n  FROZEN   codec: max single-tick drop {fd,5:P0}, collapses (>10pt) {fc}");
        Console.WriteLine($"  UNFROZEN codec: max single-tick drop {ud,5:P0}, collapses (>10pt) {uc}");
        Console.WriteLine("\n  If frozen stays smooth while unfrozen periodically collapses, the codec-pinning (RestoreFrozen) is exactly what");
        Console.WriteLine("  keeps the averaging from damaging the holographic arithmetic — which is why the live swarm never collapses.");
    }
}
