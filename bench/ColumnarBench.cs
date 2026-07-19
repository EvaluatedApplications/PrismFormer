// Copyright (c) 2026 Dongyang Stephen Chen, trading as Evaluated Applications. All rights reserved.
// Source-available for study and verification; commercial use, redistribution, or derivatives require permission. See LICENSE.

using PrismFormer;

namespace PrismFormer.Bench;

/// <summary>
/// WORKED-OUT arithmetic, each operation trained IN ISOLATION (run: <c>dotnet run --project bench -c Release -- --columnar</c>).
///
/// A sequence model cannot emit an exact multi-digit result in one token, so the fair test is to give both models
/// the same worked problem and see who learns the algorithm. We pose it digit-by-digit, reversed (least-significant
/// first) so carries flow causally, and the model must generate the answer digits itself. Each of the four
/// operations trains its OWN model (no shared budget), and all of them plus their transformer baselines train in
/// parallel. We use the ATOMIC single-pass columnar form of each: add and subtract are n±n (carry / borrow);
/// multiply is n × a single digit and divide is n ÷ a single digit, so each answer digit is again one local
/// column step (aᵢ·b + carry on the log-phase band; the linear band for ±). Full multi-digit × multi-digit needs a
/// partial-product scratchpad and is left to future work. We train on operands over 0..99, hold out ~20% of pairs
/// by a fixed hash (never trained), and measure exact-match on the unseen pairs against a parameter-matched
/// transformer, with a seen-distribution control (held-out ≈ seen ⇒ a learned algorithm, not a memorised table).
/// For PrismFormer we also decode the face at each answer digit to check its working is legible from the inside.
/// </summary>
public static class ColumnarBench
{
    const int Plus = 10, Minus = 11, Star = 12, Slash = 13, Eq = 14, End = 15, V = 16;

    static double[] Seed(int w) =>
        w <= 9 ? PhasorCodec.NumberFace(w)
        : w == Plus ? PhasorCodec.Encode("+")
        : w == Minus ? PhasorCodec.Encode("-")
        : w == Star ? PhasorCodec.Encode("*")
        : w == Slash ? PhasorCodec.Encode("/")
        : w == Eq ? PhasorCodec.Encode("=")
        : w == End ? PhasorCodec.Encode(".")
        : new double[PhasorCodec.Dim];

    static bool Held(long a, long b) { unchecked { ulong h = (ulong)(a * 1000003L) ^ (ulong)b; h *= 1099511628211UL; h ^= h >> 29; return h % 5 == 2; } }

    static (long a, long b) SampleAdd(Random r, bool held) { while (true) { long a = r.Next(0, 100), b = r.Next(0, 100); if (Held(a, b) == held) return (a, b); } }
    static (long a, long b) SampleSub(Random r, bool held) { while (true) { long a = r.Next(0, 100), b = r.Next(0, 100); if (a < b) (a, b) = (b, a); if (Held(a, b) == held) return (a, b); } }
    static (long a, long b) SampleMul(Random r, bool held) { while (true) { long a = r.Next(0, 100), b = r.Next(0, 10); if (Held(a, b) == held) return (a, b); } }   // single-digit multiplier -> single-pass columnar
    static (long a, long b) SampleDiv(Random r, bool held) { while (true) { long b = r.Next(1, 10), q = r.Next(0, 100), a = b * q; if (Held(a, b) == held) return (a, b); } }  // single-digit divisor, exact

    sealed record Op(string Name, int Sym, string Band, Func<long, long, long> R, Func<Random, bool, (long, long)> Sample);
    static readonly Op[] Ops =
    {
        new Op("add", Plus,  "linear", (a, b) => a + b, SampleAdd),
        new Op("sub", Minus, "linear", (a, b) => a - b, SampleSub),
        new Op("mul", Star,  "log",    (a, b) => a * b, SampleMul),   // n × single digit
        new Op("div", Slash, "log",    (a, b) => a / b, SampleDiv),   // n ÷ single digit, exact
    };

    static int[] DigitsLsb(long n) { if (n == 0) return new[] { 0 }; var d = new List<int>(); while (n > 0) { d.Add((int)(n % 10)); n /= 10; } return d.ToArray(); }
    static int[] Problem(long a, long b, Op op) { var s = new List<int>(); s.AddRange(DigitsLsb(a)); s.Add(op.Sym); s.AddRange(DigitsLsb(b)); s.Add(Eq); return s.ToArray(); }
    static int[] Full(long a, long b, Op op) { var s = new List<int>(Problem(a, b, op)); s.AddRange(DigitsLsb(op.R(a, b))); s.Add(End); return s.ToArray(); }

    public static void Run(int seeds = 4, int steps = 15000, bool tuned = false)
    {
        const int ctx = 20;
        var evalN = 300;

        // TUNED baseline (--tuned-baseline): the transformer gets the modern recipe — pre-norm LayerNorm, a
        // linear-warmup→cosine-decay LR schedule, and an explicit tuned Adam (β1 .9, β2 .999, ε 1e-8). AlgFormer's
        // own training (constant 1e-3) is untouched, so this only STRENGTHENS the baseline. Same warmup→cosine shape
        // as MultiTaskBench, expressed over training STEPS here rather than epochs. Peak 2e-3 (a warmup-protected
        // peak the AlgFormer already tolerates without divergence for a pre-norm stack).
        var warm = Math.Max(100, steps / 20);
        double LrTuned(int i) { const double peak = 2e-3; if (i < warm) return peak * (i + 1) / warm; var t = (i - warm) / (double)Math.Max(1, steps - warm); return peak * (0.05 + 0.95 * 0.5 * (1 + Math.Cos(Math.PI * t))); }

        Console.WriteLine($"PrismFormer worked-arithmetic probe — each operation in isolation, show your working{(tuned ? "   [TUNED transformer baseline: pre-norm LayerNorm + warmup→cosine + tuned Adam]" : "")}\n");
        Console.WriteLine($"operands 0..99  ·  multiply/divide by a single digit (single-pass columnar)  ·  ~20% of pairs held out  ·  {seeds} seeds × {steps} steps per operation\n");

        if (tuned)
        {
            if (!AlgFormer.GradCheck(out var ar) || !MiniTransformer.GradCheck(out var xr, layerNorm: true)) { Console.WriteLine("GRADCHECK FAILED — aborting"); return; }
            Console.WriteLine($"gradchecks pass (prism {ar:E1}, transformer +LN {xr:E1})\n");
        }

        var algs = new AlgFormer[Ops.Length][]; var xfs = new MiniTransformer[Ops.Length][]; var algsU = new AlgFormer[Ops.Length][];
        for (var oi = 0; oi < Ops.Length; oi++) { algs[oi] = new AlgFormer[seeds]; xfs[oi] = new MiniTransformer[seeds]; algsU[oi] = new AlgFormer[seeds]; }

        // one training job per (operation, seed, model kind); each model sees ONLY its own operation, full budget; all in parallel.
        // kind 0 = PrismFormer (frozen identity), 1 = transformer, 2 = PrismFormer ABLATION (frozenPrefix=0 → identity learnable).
        var jobs = new List<(int oi, int seed, int kind)>();
        for (var oi = 0; oi < Ops.Length; oi++) for (var s = 0; s < seeds; s++) { jobs.Add((oi, s, 0)); jobs.Add((oi, s, 1)); jobs.Add((oi, s, 2)); }

        Parallel.ForEach(jobs, job =>
        {
            var op = Ops[job.oi]; var sd = job.oi * 1000 + job.seed;
            void TrainOne(Action<int[], int, int> step)
            {
                var r = new Random(200 + sd);
                for (var i = 0; i < steps; i++)
                {
                    var (a, b) = op.Sample(r, false);
                    var full = Full(a, b, op); var start = Problem(a, b, op).Length;
                    for (var t = start; t < full.Length; t++) step(full[..t], full[t], i);
                }
            }
            if (job.kind == 1)
            {
                // TUNED: pre-norm LayerNorm transformer trained on warmup→cosine + tuned Adam; else the original
                // no-LayerNorm baseline at a constant 1e-3. Same fixed shape (104/208/4) either way → param-matched.
                var m = new MiniTransformer(vocab: V, dModel: 104, dff: 208, layers: 4, maxT: ctx, seed: 300 + sd, layerNorm: tuned);
                if (tuned) TrainOne((t, a, i) => m.TrainStep(t, a, LrTuned(i), 0.9, 0.999, 1e-8));
                else       TrainOne((t, a, i) => m.TrainStep(t, a, 1e-3));
                xfs[job.oi][job.seed] = m;
            }
            else
            {
                var frozen = job.kind == 0 ? PhasorCodec.FrozenReals : 0;
                var m = new AlgFormer(vocab: V, shifts: 48, layers: 4, maxContext: ctx,
                                      dModel: PhasorCodec.Dim, frozenPrefix: frozen, embedSeed: Seed, seed: 100 + sd);
                TrainOne((t, a, i) => m.TrainStep(t, a, 1e-3));
                if (job.kind == 0) algs[job.oi][job.seed] = m; else algsU[job.oi][job.seed] = m;
            }
        });

        Console.WriteLine($"params: PrismFormer {algs[0][0].ParamCount:N0}   transformer {xfs[0][0].ParamCount:N0}{(tuned ? " (+LayerNorm)" : "")}\n");
        Console.WriteLine("exact-match on UNSEEN held-out pairs (mean ± sd)   [seen-distribution control]:");
        var frozenHeld = new double[Ops.Length][]; var unfrozenHeld = new double[Ops.Length][];
        for (var oi = 0; oi < Ops.Length; oi++)
        {
            var op = Ops[oi];
            double[] ah = new double[seeds], xh = new double[seeds], an = new double[seeds], xn = new double[seeds], sdd = new double[seeds], uh = new double[seeds];
            Parallel.For(0, seeds, s =>
            {
                ah[s] = EvalExact(algs[oi][s].Predict, op, evalN, new Random(900 + s), true, ctx);
                xh[s] = EvalExact(xfs[oi][s].Predict, op, evalN, new Random(900 + s), true, ctx);
                an[s] = EvalExact(algs[oi][s].Predict, op, evalN, new Random(920 + s), false, ctx);
                xn[s] = EvalExact(xfs[oi][s].Predict, op, evalN, new Random(920 + s), false, ctx);
                sdd[s] = StepDecode(algs[oi][s], op, evalN, new Random(970 + s), true);
                uh[s] = EvalExact(algsU[oi][s].Predict, op, evalN, new Random(900 + s), true, ctx);
            });
            frozenHeld[oi] = ah; unfrozenHeld[oi] = uh;
            var (AH, AHs) = MS(ah); var (XH, XHs) = MS(xh); var (AN, _) = MS(an); var (XN, __) = MS(xn); var (SD, _3) = MS(sdd);
            Console.WriteLine($"  {op.Name} ({op.Band}):   PrismFormer {AH,6:P1} ± {AHs:P1}  [{AN:P0}]     transformer {XH,6:P1} ± {XHs:P1}  [{XN:P0}]     face-decode {SD:P0}");
        }
        Console.WriteLine("\nablation — frozen numeric identity vs learnable (frozenPrefix=0), held-out exact-match:");
        for (var oi = 0; oi < Ops.Length; oi++)
        {
            var (f, fs) = MS(frozenHeld[oi]); var (u, us) = MS(unfrozenHeld[oi]);
            Console.WriteLine($"  {Ops[oi].Name}:  frozen {f,6:P1} ± {fs:P1}   unfrozen {u,6:P1} ± {us:P1}");
        }
        Console.WriteLine("\n[seen] = accuracy on trained pairs; held-out ~ seen ⇒ a learned algorithm, not a table.");
        Console.WriteLine("mul/div use a single-digit second operand (single-pass columnar); full multi-digit × multi-digit needs a partial-product scratchpad.");
    }

    static double EvalExact(Func<int[], int> predict, Op op, int n, Random r, bool wantHeld, int cap)
    {
        var ok = 0;
        for (var i = 0; i < n; i++)
        {
            var (a, b) = op.Sample(r, wantHeld);
            var gen = new List<int>(Problem(a, b, op)); var digits = new List<int>();
            while (gen.Count < cap) { int t = predict(gen.ToArray()); if (t == End || t > 9) break; digits.Add(t); gen.Add(t); }
            long val = 0, mul = 1; foreach (var d in digits) { val += d * mul; mul *= 10; }
            if (digits.Count > 0 && val == op.R(a, b)) ok++;
        }
        return ok / (double)n;
    }

    static double StepDecode(AlgFormer alg, Op op, int n, Random r, bool wantHeld)
    {
        int ok = 0, tot = 0;
        for (var i = 0; i < n; i++)
        {
            var (a, b) = op.Sample(r, wantHeld);
            var full = Full(a, b, op); var start = Problem(a, b, op).Length;
            for (var t = start; t < full.Length; t++)
            {
                if (full[t] > 9) continue;
                var face = alg.LayerFaces(full[..t])[^1];
                if (PhasorCodec.DecodeSum(face, 9) == full[t]) ok++;
                tot++;
            }
        }
        return tot > 0 ? ok / (double)tot : 0;
    }

    static (double mean, double sd) MS(IReadOnlyList<double> xs)
    {
        var m = xs.Average();
        var sd = xs.Count > 1 ? Math.Sqrt(xs.Sum(x => (x - m) * (x - m)) / (xs.Count - 1)) : 0.0;
        return (m, sd);
    }
}
