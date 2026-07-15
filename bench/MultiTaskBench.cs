// Copyright (c) 2026 Dongyang Stephen Chen, trading as Evaluated Applications. All rights reserved.
// Source-available for study and verification; commercial use, redistribution, or derivatives require permission. See LICENSE.

using PrismFormer;

namespace PrismFormer.Bench;

/// <summary>
/// Multi-task generalisation, seeded (default run, or <c>--multitask</c>). Both models share vocab, context, train
/// order, LR schedule and parameter budget (the transformer's shape is auto-searched to match PrismFormer's params).
/// We report per-task <b>train AND held-out</b> accuracy as mean ± sd over N seeds, so the reader can see the baseline
/// is a real (fitting) model, not a strawman. Tasks are split into two honest groups:
///   • COMPARISON / RELATIONAL (fair): gt, max, min, parity, copy, and reverse-inference antonym/capital — the answer
///     is NOT a trivial algebraic function of the input faces, so this tests whether the model learned the relation.
///   • SINGLE-TOKEN ARITHMETIC (representation-favoured): add, sub, mul, div — the codec makes the answer face equal
///     to bind(inputs), so PrismFormer can decode it by construction. We keep it visible but it is NOT the arithmetic
///     result; the fair arithmetic test is the worked-columnar one (ColumnarBench, --columnar).
/// </summary>
internal static class MultiTaskBench
{
    internal readonly record struct Inst(string Task, string[] Prompt, string Target);

    public static void Run(int seeds = 5, int epochs = 150, int hi = 12)
    {
        try { Console.OutputEncoding = System.Text.Encoding.UTF8; } catch { }

        // ---- data (fixed across seeds; only model init + train order vary) ----
        var rng = new Random(7);
        var all = new List<Inst>();
        void Emit(string task, string[] p, object t) => all.Add(new Inst(task, p, t.ToString()!));
        for (var a = 0; a <= hi; a++)
            for (var b = 0; b <= hi; b++)
            {
                Emit("add", new[] { $"{a}", "+", $"{b}", "=" }, a + b);
                if (a >= b) Emit("sub", new[] { $"{a}", "-", $"{b}", "=" }, a - b);
                if (a >= 1 && b >= 1) { Emit("mul", new[] { $"{a}", "*", $"{b}", "=" }, a * b); Emit("div", new[] { $"{a * b}", "/", $"{b}", "=" }, a); }
                Emit("gt", new[] { $"{a}", ">", $"{b}", "=" }, a > b ? "yes" : "no");
                Emit("max", new[] { "max", $"{a}", $"{b}", "=" }, Math.Max(a, b));
                Emit("min", new[] { "min", $"{a}", $"{b}", "=" }, Math.Min(a, b));
            }
        for (var n = 0; n <= 49; n++) Emit("parity", new[] { $"{n}", "is", "=" }, n % 2 == 0 ? "even" : "odd");
        string[] W = { "red", "blue", "green", "dog", "cat", "sun", "moon", "tree", "fish", "gold", "star", "leaf" };
        var seenTriple = new HashSet<string>();
        for (var n = 0; n < 260 && seenTriple.Count < 220; n++)
        {
            string w0 = W[rng.Next(W.Length)], w1 = W[rng.Next(W.Length)], w2 = W[rng.Next(W.Length)];
            if (!seenTriple.Add($"{w0} {w1} {w2}")) continue;
            Emit("copy", new[] { "first", "of", w0, w1, w2, ":" }, w0);
        }
        var capitals = new[] { ("france", "paris"), ("england", "london"), ("germany", "berlin"), ("japan", "tokyo"), ("spain", "madrid"), ("italy", "rome"), ("portugal", "lisbon"), ("austria", "vienna"), ("greece", "athens"), ("norway", "oslo"), ("ireland", "dublin"), ("egypt", "cairo"), ("canada", "ottawa"), ("russia", "moscow"), ("china", "beijing"), ("thailand", "bangkok"), ("cuba", "havana"), ("peru", "lima"), ("poland", "warsaw"), ("sweden", "stockholm"), ("finland", "helsinki"), ("hungary", "budapest"), ("denmark", "copenhagen"), ("turkey", "ankara") };
        var antonyms = new[] { ("up", "down"), ("left", "right"), ("day", "night"), ("open", "shut"), ("win", "lose"), ("buy", "sell"), ("push", "pull"), ("give", "take"), ("love", "hate"), ("war", "peace"), ("north", "south"), ("east", "west"), ("king", "queen"), ("front", "back"), ("top", "bottom"), ("friend", "enemy"), ("rise", "fall"), ("accept", "reject"), ("arrive", "depart"), ("remember", "forget"), ("enter", "exit"), ("sink", "float"), ("laugh", "cry"), ("import", "export") };

        var fairTasks = new[] { "gt", "max", "min", "parity", "copy", "antonym", "capital" };
        var arithTasks = new[] { "add", "sub", "mul", "div" };

        static int Bucket(Inst x) { uint h = 2166136261; foreach (var c in $"{x.Task}|{string.Join(' ', x.Prompt)}={x.Target}") { h ^= c; h *= 16777619; } return (int)(h % 100); }
        var train = all.Where(x => Bucket(x) >= 20).ToList();
        var held = all.Where(x => Bucket(x) < 20).ToList();
        foreach (var (a, b) in antonyms)
        {
            train.Add(new Inst("antonym", new[] { "opposite", "of", a, "=" }, b));
            held.Add(new Inst("antonym", new[] { "opposite", "of", b, "=" }, a));
        }
        foreach (var (c, city) in capitals)
        {
            train.Add(new Inst("capital", new[] { "capital", "of", c, "=" }, city));
            held.Add(new Inst("capital", new[] { city, "is", "the", "capital", "of", "=" }, c));
        }

        // ---- shared vocab ----
        var id = new Dictionary<string, int>(StringComparer.Ordinal) { ["<pad>"] = 0 };
        var words = new List<string> { "<pad>" };
        int Id(string w) { if (id.TryGetValue(w, out var i)) return i; i = id.Count; id[w] = i; words.Add(w); return i; }
        foreach (var ins in train.Concat(held)) { foreach (var t in ins.Prompt) Id(t); Id(ins.Target); }
        var vocab = Math.Max(1024, id.Count + 8);
        (int[] Ctx, int Target) Tok(Inst x) => (x.Prompt.Select(Id).ToArray(), Id(x.Target));
        var trainPairs = train.Select(Tok).ToList();

        double[] Seed(int w) => w < words.Count ? PhasorCodec.Encode(words[w]) : new double[PhasorCodec.Dim];
        var probe = new AlgFormer(vocab, shifts: 8, layers: 2, maxContext: 16, dModel: PhasorCodec.Dim, frozenPrefix: PhasorCodec.FrozenReals, embedSeed: Seed, seed: 1);
        var targetParams = probe.ParamCount;
        // Restrict to trainable depths (L<=4): MiniTransformer has no layer-norm, so a deep-narrow stack is barely
        // trainable at this LR. Matching params by WIDTH (larger d/ff) at shallow depth gives a fair, fitting baseline.
        (int d, int L, int ff) best = (32, 2, 64); var bestDelta = long.MaxValue;
        foreach (var d in new[] { 24, 32, 40, 48, 56, 64, 96, 128, 160, 192, 256 })
            foreach (var L in new[] { 2, 3, 4 })
                foreach (var ff in new[] { 32, 48, 64, 96, 128, 160, 192, 256, 384, 512, 768, 1024 })
                { var pr = new MiniTransformer(vocab, d, ff, L, 16, 42); var delta = Math.Abs(pr.ParamCount - targetParams); if (delta < bestDelta) { bestDelta = delta; best = (d, L, ff); } }

        Console.WriteLine($"PrismFormer vs pound-for-pound transformer — multi-task generalisation   {DateTime.Now:yyyy-MM-dd HH:mm}");
        if (!AlgFormer.GradCheck(out var ar) || !MiniTransformer.GradCheck(out var xr)) { Console.WriteLine("GRADCHECK FAILED — aborting"); return; }
        Console.WriteLine($"gradchecks pass (prism {ar:E1}, transformer {xr:E1})");
        Console.WriteLine($"params: transformer {new MiniTransformer(vocab, best.d, best.ff, best.L, 16, 42).ParamCount:N0} (d={best.d}, dff={best.ff}, L={best.L})   prism {targetParams:N0} (d={PhasorCodec.Dim}, S=8, L=2)");
        Console.WriteLine($"train {train.Count:N0}   held-out {held.Count:N0}   vocab {id.Count}   {seeds} seeds × {epochs} epochs\n");

        double Acc(Func<int[], int> predict, List<Inst> set, string task)
        {
            var s = set.Where(x => x.Task == task).ToList();
            if (s.Count == 0) return double.NaN;
            var ok = 0; foreach (var x in s) { var (c, t) = Tok(x); if (predict(c) == t) ok++; }
            return ok / (double)s.Count;
        }

        var allTasks = fairTasks.Concat(arithTasks).ToArray();
        // [seed][task] -> acc
        var xfTr = new Dictionary<string, double[]>(); var xfHe = new Dictionary<string, double[]>();
        var alTr = new Dictionary<string, double[]>(); var alHe = new Dictionary<string, double[]>();
        // unfrozen-identity ablation (frozenPrefix:0), same corpus / init seed / train order as the frozen model,
        // so any difference is purely the frozen-vs-learnable numeric identity under SHARED multi-task load.
        var auTr = new Dictionary<string, double[]>(); var auHe = new Dictionary<string, double[]>();
        foreach (var t in allTasks) { xfTr[t] = new double[seeds]; xfHe[t] = new double[seeds]; alTr[t] = new double[seeds]; alHe[t] = new double[seeds]; auTr[t] = new double[seeds]; auHe[t] = new double[seeds]; }

        Parallel.For(0, seeds, s =>
        {
            var alg = new AlgFormer(vocab, shifts: 8, layers: 2, maxContext: 16, dModel: PhasorCodec.Dim, frozenPrefix: PhasorCodec.FrozenReals, embedSeed: Seed, seed: 1 + s);
            var algU = new AlgFormer(vocab, shifts: 8, layers: 2, maxContext: 16, dModel: PhasorCodec.Dim, frozenPrefix: 0, embedSeed: Seed, seed: 1 + s);
            var xf = new MiniTransformer(vocab, best.d, best.ff, best.L, 16, 42 + s);
            for (var ep = 1; ep <= epochs; ep++)
            {
                var lr = 2e-3 * (1.0 - 0.9 * (ep - 1) / Math.Max(1, epochs - 1));
                var order = Enumerable.Range(0, trainPairs.Count).ToArray();
                var er = new Random(1000 + ep + 97 * s);
                for (var i = order.Length - 1; i > 0; i--) { var j = er.Next(i + 1); (order[i], order[j]) = (order[j], order[i]); }
                foreach (var idx in order) { var (c, t) = trainPairs[idx]; alg.TrainStep(c, t, lr); }
                foreach (var idx in order) { var (c, t) = trainPairs[idx]; algU.TrainStep(c, t, lr); }
                foreach (var idx in order) { var (c, t) = trainPairs[idx]; xf.TrainStep(c, t, lr); }
            }
            foreach (var t in allTasks)
            {
                xfTr[t][s] = Acc(xf.Predict, train, t); xfHe[t][s] = Acc(xf.Predict, held, t);
                alTr[t][s] = Acc(alg.Predict, train, t); alHe[t][s] = Acc(alg.Predict, held, t);
                auTr[t][s] = Acc(algU.Predict, train, t); auHe[t][s] = Acc(algU.Predict, held, t);
            }
        });

        (double m, double sd) MS(double[] a) { var m = a.Average(); return (m, a.Length > 1 ? Math.Sqrt(a.Sum(x => (x - m) * (x - m)) / (a.Length - 1)) : 0); }
        string Cell(double[] tr, double[] he) { var (mt, _) = MS(tr); var (mh, sh) = MS(he); return $"{mt,5:P0} / {mh,5:P0}±{sh:P0}"; }

        Console.WriteLine("per-task  train / HELD-OUT (mean ± sd over seeds) ------------------------------------");
        Console.WriteLine($"  {"task",-9} {"transformer (train/held)",-24} {"prismformer (train/held)",-24}");
        Console.WriteLine("  -- comparison / relational (fair: answer is not bind(inputs)) --");
        foreach (var t in fairTasks) Console.WriteLine($"  {t,-9} {Cell(xfTr[t], xfHe[t]),-24} {Cell(alTr[t], alHe[t]),-24}");
        Console.WriteLine("  -- single-token arithmetic (representation-favoured; fair test is --columnar) --");
        foreach (var t in arithTasks) Console.WriteLine($"  {t,-9} {Cell(xfTr[t], xfHe[t]),-24} {Cell(alTr[t], alHe[t]),-24}");

        double GroupHeld(Dictionary<string, double[]> he, string[] ts) => ts.SelectMany(t => he[t]).Average();
        double GroupTrain(Dictionary<string, double[]> tr, string[] ts) => ts.SelectMany(t => tr[t]).Average();
        Console.WriteLine("\nsummary (mean over fair tasks & seeds) ----------------------------------------------");
        Console.WriteLine($"  fair tasks  train    : transformer {GroupTrain(xfTr, fairTasks):P1}   prismformer {GroupTrain(alTr, fairTasks):P1}");
        Console.WriteLine($"  fair tasks  held-out : transformer {GroupHeld(xfHe, fairTasks):P1}   prismformer {GroupHeld(alHe, fairTasks):P1}");
        Console.WriteLine($"  (transformer train accuracy shows it IS a fitting baseline, not a strawman.)");

        // ---- frozen vs unfrozen identity, under SHARED multi-task load ----
        // In per-op isolation (--columnar) unfrozen wins because it can specialise the pass-through to one operation.
        // Here the SAME identity is the residual for every task at once; the question is whether the unfrozen win survives.
        Console.WriteLine("\nfrozen vs unfrozen numeric identity — SHARED-corpus held-out (mean ± sd over seeds) --");
        Console.WriteLine($"  {"task",-9} {"frozen (train/held)",-24} {"unfrozen (train/held)",-24}");
        foreach (var t in allTasks) Console.WriteLine($"  {t,-9} {Cell(alTr[t], alHe[t]),-24} {Cell(auTr[t], auHe[t]),-24}");
        Console.WriteLine($"  fair    held-out : frozen {GroupHeld(alHe, fairTasks):P1}   unfrozen {GroupHeld(auHe, fairTasks):P1}");
        Console.WriteLine($"  arith   held-out : frozen {GroupHeld(alHe, arithTasks):P1}   unfrozen {GroupHeld(auHe, arithTasks):P1}");
        Console.WriteLine($"  ALL     held-out : frozen {GroupHeld(alHe, allTasks):P1}   unfrozen {GroupHeld(auHe, allTasks):P1}");
    }
}
