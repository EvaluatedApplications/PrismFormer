// Copyright (c) 2026 Dongyang Stephen Chen, trading as Evaluated Applications. All rights reserved.
// Source-available for study and verification; commercial use, redistribution, or derivatives require permission. See LICENSE.

using PrismFormer;

namespace PrismFormer.Bench;

/// <summary>
/// CODEC-SEEDED BASELINE CONTROL (--codec-baseline) — resolves the sharpest confound named in paper 1 §6-A:
/// PrismFormer's number embeddings are seeded from the phasor codec while the transformer baseline starts random, so
/// part of the measured gap could be INITIALISATION, not architecture. This gives the transformer the SAME seeding and
/// re-measures. Three models on held-out (in-range, unseen-combination) add/sub: (1) AlgFormer, codec-seeded; (2) a
/// MiniTransformer with RANDOM embeddings; (3) the SAME MiniTransformer with embeddings seeded from the codec
/// (identical params to #2 — the only difference is init). If seeding lifts the transformer toward AlgFormer, the gap
/// was partly init; if AlgFormer still leads a seeded transformer, the architecture is doing real work.
/// </summary>
internal static class BaselineControlBench
{
    public static void Run(int epochs = 100)
    {
        try { Console.OutputEncoding = System.Text.Encoding.UTF8; } catch { }
        const int HI = 12;   // operands 0..12; held-out = 20% of in-range combinations (unseen operand pairs)
        var rng = new Random(7);
        var train = new List<(int a, int b, char op, int c)>();
        var held = new List<(int a, int b, char op, int c)>();
        for (var a = 0; a <= HI; a++)
            for (var b = 0; b <= HI; b++)
                foreach (var op in new[] { '+', '-' })
                {
                    if (op == '-' && a < b) continue;
                    var x = (a, b, op, op == '+' ? a + b : a - b);
                    if (rng.NextDouble() < 0.2) held.Add(x); else train.Add(x);
                }

        var id = new Dictionary<string, int>(StringComparer.Ordinal) { ["<pad>"] = 0 };
        var words = new List<string> { "<pad>" };
        int Id(string w) { if (id.TryGetValue(w, out var i)) return i; i = id.Count; id[w] = i; words.Add(w); return i; }
        for (var n = 0; n <= 2 * HI; n++) Id($"{n}"); Id("+"); Id("-"); Id("=");
        var vocab = Math.Max(32, id.Count);
        int[] Ctx((int a, int b, char op, int c) x) => new[] { Id($"{x.a}"), Id(x.op.ToString()), Id($"{x.b}"), Id("=") };
        int Tgt((int a, int b, char op, int c) x) => Id($"{x.c}");

        // seed: numbers -> homomorphic NumberFace, symbols -> orthogonal signature (same for AlgFormer + the seeded transformer)
        double[] Seed(int w)
        {
            if (w >= words.Count) return new double[PhasorCodec.Dim];
            return int.TryParse(words[w], out var n) ? PhasorCodec.NumberFace(n) : PhasorCodec.Encode(words[w]);
        }

        var alg = AlgFormer.Mini(vocab, embedSeed: Seed, seed: 1);
        var mtRand = new MiniTransformer(vocab, dModel: PhasorCodec.Dim, dff: 256, layers: 2, maxT: 4, seed: 1);
        var mtSeed = new MiniTransformer(vocab, dModel: PhasorCodec.Dim, dff: 256, layers: 2, maxT: 4, seed: 1);
        for (var w = 0; w < words.Count; w++) { var s = Seed(w); Array.Copy(s, mtSeed.Emb[w], PhasorCodec.Dim); }   // ONLY difference from mtRand: seeded embeddings

        Console.WriteLine($"CODEC-SEEDED BASELINE CONTROL — held-out arithmetic (add/sub 0..{HI}, unseen operand pairs)   {DateTime.Now:yyyy-MM-dd HH:mm}");
        Console.WriteLine($"  params:  AlgFormer {alg.ParamCount:N0}   MiniTransformer {mtRand.ParamCount:N0} (rand & seeded identical)");
        Console.WriteLine($"  train {train.Count} / held-out {held.Count}\n");

        double Acc(Func<int[], int> p, List<(int a, int b, char op, int c)> set) { var ok = 0; foreach (var x in set) if (p(Ctx(x)) == Tgt(x)) ok++; return ok / (double)set.Count; }
        var tp = train.Select(x => (Ctx(x), Tgt(x))).ToList();

        for (var ep = 1; ep <= epochs; ep++)
        {
            var lr = 2e-3 * (1.0 - 0.9 * (ep - 1) / Math.Max(1, epochs - 1));
            var order = Enumerable.Range(0, tp.Count).OrderBy(_ => rng.Next()).ToArray();
            System.Threading.Tasks.Parallel.Invoke(
                () => { foreach (var i in order) { var (c, t) = tp[i]; alg.TrainStep(c, t, lr); } },
                () => { foreach (var i in order) { var (c, t) = tp[i]; mtRand.TrainStep(c, t, lr); } },
                () => { foreach (var i in order) { var (c, t) = tp[i]; mtSeed.TrainStep(c, t, lr); } });
            if (ep % 20 == 0 || ep == epochs)
                Console.WriteLine($"  epoch {ep,3}/{epochs}:  AlgFormer {Acc(alg.Predict, held),5:P0}   transformer(random) {Acc(mtRand.Predict, held),5:P0}   transformer(codec-seeded) {Acc(mtSeed.Predict, held),5:P0}");
        }

        var a0 = Acc(alg.Predict, held); var r0 = Acc(mtRand.Predict, held); var s0 = Acc(mtSeed.Predict, held);
        var aT = Acc(alg.Predict, train); var rT = Acc(mtRand.Predict, train); var sT = Acc(mtSeed.Predict, train);
        Console.WriteLine("\n  RESULT (train acc shows they FIT; held-out acc shows they GENERALISE):");
        Console.WriteLine($"    {"model",-28} {"train",8} {"held-out",10}");
        Console.WriteLine($"    {"AlgFormer (codec-seeded)",-28} {aT,7:P0} {a0,9:P1}");
        Console.WriteLine($"    {"transformer (random init)",-28} {rT,7:P0} {r0,9:P1}");
        Console.WriteLine($"    {"transformer (codec-seeded)",-28} {sT,7:P0} {s0,9:P1}   <- the control");
        Console.WriteLine($"\n  seeding lift on the transformer: {(s0 - r0 >= 0 ? "+" : "")}{s0 - r0:P1}. Gap AlgFormer still holds over a");
        Console.WriteLine($"  SEEDED transformer: {(a0 - s0 >= 0 ? "+" : "")}{a0 - s0:P1}. If that residual is large, the advantage is architecture, not just init.");
    }
}
