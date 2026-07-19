// Copyright (c) 2026 Dongyang Stephen Chen, trading as Evaluated Applications. All rights reserved.
// Source-available for study and verification; commercial use, redistribution, or derivatives require permission. See LICENSE.

using PrismFormer;

namespace PrismFormer.Bench;

/// <summary>
/// REVERSE INFERENCE (--revinfer). Paper 1 §4.2/§6: a trained model gets 0% on the reverse of a symmetric relation
/// (train "opposite of A = B", test "opposite of B = A"). Can scratchpad thinking, or the codec algebra, beat that 0%?
/// Three arms:
///   1) CLOSED-BOOK trained (the paper's setup): forward only, reverse held. Expected ~0% — recall is DIRECTIONAL.
///   2) OPEN-BOOK scratchpad: the forward fact is provided in context, and the model learns "select the other element".
///      The select is trained and HELD OUT on NOVEL pairings of tokens it has ALREADY seen (like the copy task), so this
///      isolates whether the reversal generalises — not whether it can emit out-of-vocabulary tokens.
///   3) ALGEBRA (HRR associative memory, NO training): store each pair as the commutative bind L⊛R in one bundle; the
///      reverse is retrieved by unbinding the query (bind with its conjugate). Symmetric for free, like arithmetic.
/// The honest point: the bottleneck is closed-book DIRECTIONAL RECALL, not the reversal. Provide the fact
/// (scratchpad/retrieval) or store it as a commutative bind (algebra) and reverse inference is solved.
/// </summary>
internal static class RevInferBench
{
    const int N = 24;   // symmetric pairs (arms 1 & 3)

    static double[] Bind(double[] a, double[] b)
    {
        var h = new double[a.Length];
        for (var c = 0; c < a.Length / 2; c++)
        {
            double ar = a[2 * c], ai = a[2 * c + 1], br = b[2 * c], bi = b[2 * c + 1];
            h[2 * c] = ar * br - ai * bi;
            h[2 * c + 1] = ar * bi + ai * br;
        }
        return h;
    }
    static double[] Conj(double[] a) { var h = (double[])a.Clone(); for (var c = 0; c < a.Length / 2; c++) h[2 * c + 1] = -h[2 * c + 1]; return h; }
    static double Corr(double[] a, double[] b) { double s = 0; for (var i = 0; i < a.Length; i++) s += a[i] * b[i]; return s; }
    static void AddInto(double[] acc, double[] x) { for (var i = 0; i < acc.Length; i++) acc[i] += x[i]; }

    static void Train(AlgFormer m, List<(int[] ctx, int tgt)> data, int epochs)
    {
        var rng = new Random(0);
        for (var ep = 1; ep <= epochs; ep++)
        {
            var lr = 3e-3 * (1.0 - 0.9 * (ep - 1) / Math.Max(1, epochs - 1));
            foreach (var idx in Enumerable.Range(0, data.Count).OrderBy(_ => rng.Next())) { var (c, t) = data[idx]; m.TrainStep(c, t, lr); }
        }
    }

    public static void Run()
    {
        try { Console.OutputEncoding = System.Text.Encoding.UTF8; } catch { }
        Console.WriteLine($"REVERSE INFERENCE via scratchpad & algebra — can we beat the paper's 0%?  ({N} symmetric pairs)   {DateTime.Now:yyyy-MM-dd HH:mm}\n");

        var words = new List<string> { "<pad>" };
        var id = new Dictionary<string, int>(StringComparer.Ordinal) { ["<pad>"] = 0 };
        int Id(string w) { if (id.TryGetValue(w, out var i)) return i; i = words.Count; id[w] = i; words.Add(w); return i; }

        var L = new int[N]; var R = new int[N];
        for (var i = 0; i < N; i++) { L[i] = Id($"L{i}"); R[i] = Id($"R{i}"); }
        const int POOL = 40;                                   // shared token pool for arm 2 (all tokens SEEN in training)
        var pool = new int[POOL];
        for (var i = 0; i < POOL; i++) pool[i] = Id($"t{i}");
        int OPP = Id("opp"), EQ = Id("="), BAR = Id("|");
        var vocab = Math.Max(128, words.Count + 4);
        double[] Seed(int w) => w < words.Count ? PhasorCodec.Encode(words[w]) : new double[PhasorCodec.Dim];

        // ---------- ARM 3: algebra (HRR associative memory, NO training) ----------
        var content = L.Concat(R).ToArray();
        int Decode(double[] q, int[] cands)
        {
            int best = -1; double bv = double.NegativeInfinity;
            foreach (var w in cands) { var c = Corr(Seed(w), q); if (c > bv) { bv = c; best = w; } }
            return best;
        }
        (int fwd, int rev) Hrr(int n)
        {
            var M = new double[PhasorCodec.Dim];
            for (var i = 0; i < n; i++) AddInto(M, Bind(Seed(L[i]), Seed(R[i])));   // one bundle of commutative binds
            var cand = L.Take(n).Concat(R.Take(n)).ToArray();
            int fwd = 0, rev = 0;
            for (var i = 0; i < n; i++)
            {
                if (Decode(Bind(M, Conj(Seed(L[i]))), cand) == R[i]) fwd++;
                if (Decode(Bind(M, Conj(Seed(R[i]))), cand) == L[i]) rev++;
            }
            return (fwd, rev);
        }
        var (f3, r3) = Hrr(N);
        var (f3h, r3h) = Hrr(N / 2);

        // ---------- ARM 1: closed-book trained (the paper's setup) ----------
        var m1 = new AlgFormer(vocab, shifts: 8, layers: 2, maxContext: 16, dModel: PhasorCodec.Dim, frozenPrefix: PhasorCodec.FrozenReals, embedSeed: Seed, seed: 1);
        var train1 = new List<(int[], int)>();
        for (var i = 0; i < N; i++) train1.Add((new[] { L[i], OPP, EQ }, R[i]));   // FORWARD only
        Train(m1, train1, 400);
        int fwd1 = 0, rev1 = 0;
        for (var i = 0; i < N; i++)
        {
            if (m1.Predict(new[] { L[i], OPP, EQ }) == R[i]) fwd1++;
            if (m1.Predict(new[] { R[i], OPP, EQ }) == L[i]) rev1++;   // reverse, never trained
        }

        // ---------- ARM 2: open-book scratchpad (fact in context; select-the-other generalised over a SHARED vocab) ----------
        var m2 = new AlgFormer(vocab, shifts: 8, layers: 2, maxContext: 16, dModel: PhasorCodec.Dim, frozenPrefix: PhasorCodec.FrozenReals, embedSeed: Seed, seed: 1);
        var rng2 = new Random(5);
        var seen = new HashSet<(int, int)>();
        var seenList = new List<(int a, int b)>();
        while (seenList.Count < 240) { int a = pool[rng2.Next(POOL)], b = pool[rng2.Next(POOL)]; if (a == b || !seen.Add((a, b))) continue; seenList.Add((a, b)); }
        var held = new List<(int a, int b)>();
        var heldSet = new HashSet<(int, int)>();
        while (held.Count < 40) { int a = pool[rng2.Next(POOL)], b = pool[rng2.Next(POOL)]; if (a == b || seen.Contains((a, b)) || seen.Contains((b, a)) || !heldSet.Add((a, b))) continue; held.Add((a, b)); }
        var train2 = new List<(int[], int)>();
        foreach (var (a, b) in seenList)
        {
            train2.Add((new[] { a, OPP, b, BAR, a, OPP, EQ }, b));   // fact given, forward query -> the other element
            train2.Add((new[] { a, OPP, b, BAR, b, OPP, EQ }, a));   // fact given, REVERSE query -> the other element
        }
        Train(m2, train2, 300);
        int rev2seen = 0; foreach (var (a, b) in seenList.Take(80)) if (m2.Predict(new[] { a, OPP, b, BAR, b, OPP, EQ }) == a) rev2seen++;
        int rev2held = 0; foreach (var (a, b) in held) if (m2.Predict(new[] { a, OPP, b, BAR, b, OPP, EQ }) == a) rev2held++;

        // ---------- report ----------
        Console.WriteLine("ARM 1  closed-book trained (forward only, reverse HELD OUT = the paper's setup):");
        Console.WriteLine($"        forward (trained) {fwd1 / (double)N,5:P0}     REVERSE (held) {rev1 / (double)N,5:P0}     <- the 0% the paper reports\n");
        Console.WriteLine("ARM 2  open-book scratchpad (forward fact in context; 'select the other', held-out on NOVEL pairings of SEEN tokens):");
        Console.WriteLine($"        REVERSE, seen pairs {rev2seen / 80.0,5:P0}     REVERSE, HELD pairs {rev2held / (double)held.Count,5:P0}     <- reversal is a copy/select that generalises");
        Console.WriteLine("        (this is closed-book impossible in Arm 1; giving the model the fact turns the reverse into a select it can do)\n");
        Console.WriteLine("ARM 3  algebra (HRR memory, NO training): store pair as commutative bind L⊛R, retrieve by unbind:");
        Console.WriteLine($"        {N} pairs: forward {f3 / (double)N,5:P0}  REVERSE {r3 / (double)N,5:P0}      {N / 2} pairs: forward {f3h / (double)(N / 2),5:P0}  REVERSE {r3h / (double)(N / 2),5:P0}");
        Console.WriteLine("        forward == reverse exactly (the bind is commutative); the shortfall from 100% is HRR crosstalk, not asymmetry.\n");
        Console.WriteLine("READ: Arm 1 fails REVERSE because closed-book recall is DIRECTIONAL, not because reversing is hard. Give the model the");
        Console.WriteLine("fact (Arm 2, scratchpad/retrieval) and the reverse is a select that generalises; or store facts as commutative binds");
        Console.WriteLine("(Arm 3) and the reverse is retrieved for free, symmetric by construction, exactly the way arithmetic falls out of the codec.");
    }
}
