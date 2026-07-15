// Copyright (c) 2026 Dongyang Stephen Chen, trading as Evaluated Applications. All rights reserved.
// Source-available for study and verification; commercial use, redistribution, or derivatives require permission. See LICENSE.

using PrismFormer;

namespace PrismFormer.Bench;

/// <summary>
/// One advanced capability in isolation (run with <c>--extrap</c>): OUT-OF-RANGE MAGNITUDE EXTRAPOLATION on arithmetic
/// — the canonical failure of transformers / LLMs. Both models train ONLY on small operands (0..TRAIN_HI); both are
/// then tested on operands far outside that range, whose answers were NEVER produced during training (though every
/// answer token exists in the shared vocabulary, so both models physically CAN emit it). PrismFormer's frozen
/// homomorphic number faces let it name a computed result it never saw as a training answer; the transformer, whose
/// embeddings for those out-of-range tokens are untrained, cannot. Pound-for-pound parameters.
/// </summary>
internal static class ExtrapolationBench
{
    public static void Run(int epochs = 120)
    {
        try { Console.OutputEncoding = System.Text.Encoding.UTF8; } catch { }
        const int TRAIN_HI = 12;    // operands seen in training: 0..12   (sums 0..24)
        const int EXTRAP_HI = 40;   // extrapolation operands: up to 40    (sums up to 80) — never seen
        const int MAXNUM = 2 * EXTRAP_HI + 2;

        var rng = new Random(7);
        var trainSet = new List<(int a, int b, char op, int c)>();
        var inHeld = new List<(int a, int b, char op, int c)>();
        var extrap = new List<(int a, int b, char op, int c)>();
        for (var a = 0; a <= EXTRAP_HI; a++)
            for (var b = 0; b <= EXTRAP_HI; b++)
                foreach (var op in new[] { '+', '-' })
                {
                    if (op == '-' && a < b) continue;
                    var t = (a, b, op, op == '+' ? a + b : a - b);
                    var inRange = a <= TRAIN_HI && b <= TRAIN_HI;
                    if (inRange) { if (rng.NextDouble() < 0.2) inHeld.Add(t); else trainSet.Add(t); }
                    else extrap.Add(t);
                }

        // ---- shared vocab: EVERY number token exists (its face / embedding is present), so both models can emit any answer ----
        var id = new Dictionary<string, int>(StringComparer.Ordinal) { ["<pad>"] = 0 };
        var words = new List<string> { "<pad>" };
        int Id(string w) { if (id.TryGetValue(w, out var i)) return i; i = id.Count; id[w] = i; words.Add(w); return i; }
        for (var n = 0; n <= MAXNUM; n++) Id($"{n}");
        Id("+"); Id("-"); Id("=");
        var vocab = Math.Max(256, id.Count + 4);
        int[] Ctx((int a, int b, char op, int c) x) => new[] { Id($"{x.a}"), Id(x.op.ToString()), Id($"{x.b}"), Id("=") };
        int Tgt((int a, int b, char op, int c) x) => Id($"{x.c}");

        double[] Seed(int w) => w < words.Count ? PhasorCodec.Encode(words[w]) : new double[PhasorCodec.Dim];
        var alg = AlgFormer.Mini(vocab, embedSeed: Seed, seed: 1);   // standard gated GLU feed-forward
        var algBind = new AlgFormer(vocab, shifts: 32, layers: 4, maxContext: 8, dModel: PhasorCodec.Dim, frozenPrefix: PhasorCodec.FrozenReals, embedSeed: Seed, seed: 1, bindFfn: true);   // complex-bind FFN: a structural bias to COMPOSE (which is magnitude-independent) rather than interpolate

        Console.WriteLine($"OUT-OF-RANGE EXTRAPOLATION - does a complex-bind FFN compose (extrapolate) where a GLU only interpolates?   {DateTime.Now:yyyy-MM-dd HH:mm}");
        if (!AlgFormer.GradCheck(out var ar) || !AlgFormer.GradCheckBind(out var br)) { Console.WriteLine("GRADCHECK FAILED"); return; }
        Console.WriteLine($"gradchecks pass (glu {ar:E1}, bind {br:E1})   params {alg.ParamCount:N0} each");
        Console.WriteLine($"train on operands 0..{TRAIN_HI} (+,-)   ->   test on operands up to {EXTRAP_HI} (answers never trained)");
        Console.WriteLine($"train {trainSet.Count} / in-range held {inHeld.Count} / extrapolation {extrap.Count}\n");

        double Acc(Func<int[], int> p, List<(int a, int b, char op, int c)> set) { if (set.Count == 0) return double.NaN; var ok = 0; foreach (var x in set) if (p(Ctx(x)) == Tgt(x)) ok++; return ok / (double)set.Count; }
        var tp = trainSet.Select(x => (Ctx(x), Tgt(x))).ToList();

        for (var ep = 1; ep <= epochs; ep++)
        {
            var lr = 2e-3 * (1.0 - 0.9 * (ep - 1) / Math.Max(1, epochs - 1));
            var order = Enumerable.Range(0, tp.Count).OrderBy(_ => rng.Next()).ToArray();
            System.Threading.Tasks.Parallel.Invoke(   // GLU and bind variants side by side
                () => { foreach (var i in order) { var (c, t) = tp[i]; alg.TrainStep(c, t, lr); } },
                () => { foreach (var i in order) { var (c, t) = tp[i]; algBind.TrainStep(c, t, lr); } });
            if (ep % 20 == 0 || ep == epochs) Console.WriteLine($"  epoch {ep,3}/{epochs}: extrapolation acc  glu {Acc(alg.Predict, extrap):P0}  bind {Acc(algBind.Predict, extrap):P0}");
        }

        Console.WriteLine("\nresults ------------------------------------------------------------------------");
        Console.WriteLine($"  in-range held-out (operands 0..{TRAIN_HI}) : GLU {Acc(alg.Predict, inHeld):P1}   bind {Acc(algBind.Predict, inHeld):P1}");
        Console.WriteLine("  EXTRAPOLATION by distance past the trained range:");
        foreach (var (lo, hi) in new[] { (13, 20), (21, 30), (31, 40) })
        {
            var bin = extrap.Where(x => Math.Max(x.a, x.b) >= lo && Math.Max(x.a, x.b) <= hi).ToList();
            Console.WriteLine($"    operands {lo,2}..{hi,2} : GLU {Acc(alg.Predict, bin),6:P0}   bind {Acc(algBind.Predict, bin),6:P0}   ({bin.Count} cases)");
        }
        Console.WriteLine($"  EXTRAPOLATION overall              : GLU {Acc(alg.Predict, extrap):P1}   bind {Acc(algBind.Predict, extrap):P1}");

        Console.WriteLine("\nsamples (operands far outside training; answer never seen in training):");
        foreach (var (a, op, b) in new[] { (27, '+', 35), (31, '+', 8), (38, '-', 19), (40, '+', 40), (33, '-', 12) })
        {
            var x = (a, b, op, op == '+' ? a + b : a - b);
            string R(Func<int[], int> p) { var w = p(Ctx(x)); return w >= 0 && w < words.Count ? words[w] : "?"; }
            Console.WriteLine($"  {a} {op} {b} = {x.Item4,-4} truth   glu: {R(alg.Predict),-4}   bind: {R(algBind.Predict),-4}");
        }
    }
}
