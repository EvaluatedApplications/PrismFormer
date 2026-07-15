// Copyright (c) 2026 Dongyang Stephen Chen, trading as Evaluated Applications. All rights reserved.
// Source-available for study and verification; commercial use, redistribution, or derivatives require permission. See LICENSE.

using PrismFormer;

namespace PrismFormer.Bench;

/// <summary>
/// Scaling test (run with <c>--scale</c>): does the architecture use extra capacity better than a transformer? Both
/// models are grown across several parameter budgets (PrismFormer cheaply via depth L and rank S; the transformer
/// matched in params at each size) and evaluated on held-out compute generalisation — a task where capacity, not data,
/// is the bottleneck (arithmetic/logic is a function to learn, so a bigger model should generalise better without just
/// memorising). If PrismFormer's curve stays above the transformer's and keeps rising, the architecture scales.
/// </summary>
internal static class ScaleBench
{
    public static void Run(int epochs = 80)
    {
        const int HI = 9;
        var rng = new Random(7);
        var all = new List<(string task, string[] p, int c)>();
        void E(string t, string[] p, int c) => all.Add((t, p, c));
        for (var a = 0; a <= HI; a++)
            for (var b = 0; b <= HI; b++)
            {
                E("add", new[] { $"{a}", "+", $"{b}", "=" }, a + b);
                if (a >= b) E("sub", new[] { $"{a}", "-", $"{b}", "=" }, a - b);
                if (a >= 1 && b >= 1) E("mul", new[] { $"{a}", "*", $"{b}", "=" }, a * b);
                E("max", new[] { "max", $"{a}", $"{b}", "=" }, Math.Max(a, b));
                E("min", new[] { "min", $"{a}", $"{b}", "=" }, Math.Min(a, b));
            }
        // harder, non-saturating: 3-operand composition (small models cannot master it; capacity should keep helping)
        for (var a = 0; a <= HI; a++)
            for (var b = 0; b <= HI; b++)
                for (var c = 0; c <= HI; c++)
                {
                    E("add3", new[] { $"{a}", "+", $"{b}", "+", $"{c}", "=" }, a + b + c);
                    if (a >= b + c) E("sub3", new[] { $"{a}", "-", $"{b}", "-", $"{c}", "=" }, a - b - c);
                }
        var tasks = new[] { "add", "sub", "mul", "max", "min", "add3", "sub3" };

        var id = new Dictionary<string, int>(StringComparer.Ordinal) { ["<pad>"] = 0 };
        var words = new List<string> { "<pad>" };
        int Id(string w) { if (id.TryGetValue(w, out var i)) return i; i = id.Count; id[w] = i; words.Add(w); return i; }
        for (var n = 0; n <= HI * HI; n++) Id($"{n}");
        Id("+"); Id("-"); Id("*"); Id(">"); Id("="); Id("max"); Id("min");
        var vocab = Math.Max(256, id.Count + 4);
        int[] Ctx((string task, string[] p, int c) x) => x.p.Select(Id).ToArray();
        int Tgt((string task, string[] p, int c) x) => Id($"{x.c}");

        static int Bucket((string task, string[] p, int c) x) { uint h = 2166136261; foreach (var ch in $"{x.task}|{string.Join(' ', x.p)}={x.c}") { h ^= ch; h *= 16777619; } return (int)(h % 100); }
        var train = all.Where(x => Bucket(x) >= 20).ToList();
        var held = all.Where(x => Bucket(x) < 20).ToList();
        var tp = train.Select(x => (Ctx(x), Tgt(x))).ToList();

        double[] Seed(int w) => w < words.Count ? PhasorCodec.Encode(words[w]) : new double[PhasorCodec.Dim];
        double HeldCompute(Func<int[], int> p) => tasks.Average(t => { var s = held.Where(x => x.task == t).ToList(); return s.Count == 0 ? 0 : s.Count(x => p(Ctx(x)) == Tgt(x)) / (double)s.Count; });

        Console.WriteLine($"SCALING TEST - held-out compute generalisation vs model size (pound-for-pound at each size)   {DateTime.Now:yyyy-MM-dd HH:mm}");
        Console.WriteLine($"tasks {string.Join(' ', tasks)} over operands 0..{HI}   train {train.Count} / held {held.Count}   epochs {epochs}\n");
        Console.WriteLine($"  {"prism (L,S)",-12} {"params",-9} {"xf params",-10} | held-out compute:  transformer   prismformer   (warmup + depth-scaled LR)");

        // warmup + a learning rate scaled DOWN as depth grows (deep multiplicative stacks diverge at a fixed high LR).
        // caller-side only — no library change; TrainStep already takes the LR.
        double Lr(int ep, int depth)
        {
            var warm = Math.Max(1, epochs / 12);
            var b = 1.5e-3 * Math.Min(1.0, 4.0 / depth);
            var t = ep <= warm ? (double)ep / warm : 1.0 - 0.9 * (ep - warm) / Math.Max(1, epochs - warm);
            return b * t;
        }

        foreach (var (L, S) in new[] { (2, 16), (4, 32), (6, 48), (8, 64) })
        {
            var alg = new AlgFormer(vocab, shifts: S, layers: L, maxContext: 8, dModel: PhasorCodec.Dim, frozenPrefix: PhasorCodec.FrozenReals, embedSeed: Seed, seed: 1);
            var target = alg.ParamCount;
            (int d, int Lx, int ff) best = (32, 2, 64); var bd = long.MaxValue;
            foreach (var d in new[] { 24, 32, 40, 48, 56, 64, 80, 96, 112, 128 })
                foreach (var Lx in new[] { 2, 3, 4, 6, 8 })
                    foreach (var ff in new[] { 32, 64, 96, 128, 192, 256, 384 })
                    { var pr = new MiniTransformer(vocab, d, ff, Lx, 8, 42); if (Math.Abs(pr.ParamCount - target) < bd) { bd = Math.Abs(pr.ParamCount - target); best = (d, Lx, ff); } }
            var xf = new MiniTransformer(vocab, best.d, best.ff, best.Lx, 8, 42);

            for (var ep = 1; ep <= epochs; ep++)
            {
                var lrA = Lr(ep, L); var lrX = Lr(ep, best.Lx);   // each model's LR scaled by its own depth
                var order = Enumerable.Range(0, tp.Count).OrderBy(_ => rng.Next()).ToArray();
                System.Threading.Tasks.Parallel.Invoke(
                    () => { foreach (var i in order) { var (c, t) = tp[i]; alg.TrainStep(c, t, lrA); } },
                    () => { foreach (var i in order) { var (c, t) = tp[i]; xf.TrainStep(c, t, lrX); } });
            }
            Console.WriteLine($"  {$"({L},{S})",-12} {alg.ParamCount,-9:N0} {xf.ParamCount,-10:N0} |        {HeldCompute(xf.Predict),12:P1}   {HeldCompute(alg.Predict),11:P1}");
        }
    }
}
