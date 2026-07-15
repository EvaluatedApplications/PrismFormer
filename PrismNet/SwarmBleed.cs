using PrismFormer;

// ═══════════════════════════════════════════════════════════════════════════════════════════════════════════
//  SwarmBleed — nodes that stay SEPARATE (not copies) but let skill fragments BLEED between them (see SWARM.md).
//
//  Isolated = distinct but no transfer. FedAvg = full transfer but the nodes become identical copies. This probes
//  the MIDDLE: elastic coupling — each node is pulled SOFTLY toward the group mean by a spring of strength alpha,
//  instead of hard-averaged. alpha=0 isolated, alpha=1 copies, in between = "separate but bleeding".
//
//    Part A — two nodes on disjoint skill sets, sweep alpha: watch own-skill stay high while the OTHER node's
//             skill bleeds in ABOVE CHANCE, and the two stay measurably DIFFERENT (divergence > 0).
//    Part B — a ring of K nodes, each owning ONE skill, coupled to NEIGHBOURS ONLY: does a skill permeate to a
//             node that never trained it and isn't even adjacent to its source? (transitive diffusion)
//
//  Runnable as `prismnet swarmbleed`.
// ═══════════════════════════════════════════════════════════════════════════════════════════════════════════

static class SwarmBleed
{
    const int VOCAB = 32, DATA = 4, MAXCTX = 1 + DATA, DMODEL = 48, SHIFTS = 12, LAYERS = 3, SEED = 1;
    const int ROUNDS = 900, BATCH = 64, H = 15, EVAL = 512;
    const double LR = 6e-3, CHANCE = 100.0 / VOCAB;
    static readonly string[] TaskName = { "copy0", "last", "max", "min" };

    static AlgFormer NewModel() => new(VOCAB, shifts: SHIFTS, layers: LAYERS, maxContext: MAXCTX, dModel: DMODEL, frozenPrefix: 0, embedSeed: null, seed: SEED);

    public static void Run()
    {
        Console.WriteLine("=== skill bleed WITHOUT becoming copies — elastic coupling (see SWARM.md) ===\n");
        Console.WriteLine($"skills: {string.Join(", ", TaskName)}  |  chance {CHANCE:F1}%  |  coupling every {H} rounds\n");

        // ---------- Part A: two nodes, disjoint skills, sweep coupling strength ----------
        var aSkills = new[] { 0, 1 };   // node A trains only copy0 + last
        var bSkills = new[] { 2, 3 };   // node B trains only max + min
        Console.WriteLine("PART A — node A owns {copy0,last}, node B owns {max,min}. Sweep coupling alpha:");
        Console.WriteLine($"  {"alpha",7}{"A:own",8}{"A:B-skill",11}{"B:own",8}{"B:A-skill",11}{"divergence",12}   regime");
        foreach (var alpha in new[] { 0.0, 0.15, 0.4, 1.0 })
        {
            var A = NewModel(); var B = NewModel();
            for (var r = 0; r < ROUNDS; r++)
            {
                TrainBatch(A, aSkills[r % aSkills.Length], r);
                TrainBatch(B, bSkills[r % bSkills.Length], r);
                if ((r + 1) % H == 0 && alpha > 0) PullToMean(new[] { A, B }, alpha);
            }
            double aOwn = Avg(A, aSkills), aOther = Avg(A, bSkills), bOwn = Avg(B, bSkills), bOther = Avg(B, aSkills);
            var div = Divergence(A, B);
            var regime = alpha == 0 ? "isolated (no bleed)" : div < 0.02 ? "copies" : (aOther > 2 * CHANCE / 100 && bOther > 2 * CHANCE / 100) ? "SEPARATE + BLEEDING" : "weak bleed";
            Console.WriteLine($"  {alpha,7:F2}{aOwn,8:P0}{aOther,11:P0}{bOwn,8:P0}{bOther,11:P0}{div,12:F3}   {regime}");
        }
        Console.WriteLine("  A:B-skill / B:A-skill above chance = a fragment of the OTHER node's skill bled in.");
        Console.WriteLine("  divergence > 0 = the nodes are still DIFFERENT (not copies). Middle alpha = both at once.\n");

        // ---------- Part B: ring of 4 nodes, one skill each, neighbours-only coupling ----------
        const int K = 4; const double ringAlpha = 0.35;
        Console.WriteLine($"PART B — ring of {K} nodes, node k trains ONLY skill k, couples to neighbours (k-1,k+1) only, alpha {ringAlpha}:");
        var nodes = new AlgFormer[K]; for (var k = 0; k < K; k++) nodes[k] = NewModel();
        for (var r = 0; r < ROUNDS; r++)
        {
            for (var k = 0; k < K; k++) TrainBatch(nodes[k], k, r);
            if ((r + 1) % H == 0)
                for (var k = 0; k < K; k++)   // pull each node a little toward the mean of its two ring neighbours
                {
                    var nbrMean = Average(new[] { nodes[(k + K - 1) % K], nodes[(k + 1) % K] });
                    Blend(nodes[k], nbrMean, ringAlpha);
                }
        }
        Console.WriteLine($"  {"node",6}{"owns",8}   accuracy on each skill (own skill in [brackets]):");
        for (var k = 0; k < K; k++)
        {
            Console.Write($"  {("n" + k),6}{TaskName[k],8}   ");
            for (var s = 0; s < K; s++) { var a = Avg(nodes[k], new[] { s }); Console.Write(k == s ? $"[{a,4:P0}] " : $" {a,4:P0}  "); }
            Console.WriteLine();
        }
        Console.WriteLine($"  a node scoring above chance ({CHANCE:F0}%) on a skill it NEVER trained = that skill permeated the ring to it,");
        Console.WriteLine("  including from non-adjacent sources (2 hops away) — skill fragments diffusing through the network.");
    }

    // ---- training ----
    static long Idx(int task, int round, int i) { var h = (ulong)(round * BATCH + i) * 2654435761UL + (ulong)task * 40503UL; h ^= h >> 13; return (long)(h % 4_000_000UL); }
    static void TrainBatch(AlgFormer m, int task, int round)
    {
        var g = m.NewGrads();
        for (var i = 0; i < BATCH; i++) { var (c, t) = Example(task, Idx(task, round, i)); m.Accumulate(c, t, g); }
        m.Step(g, LR, scale: BATCH);
    }
    static (int[] Ctx, int Tgt) Example(int task, long index)
    {
        var rng = new Random(unchecked(1013 * task + (int)index * 7919 + 17));
        var ctx = new int[MAXCTX]; ctx[0] = task; for (var k = 0; k < DATA; k++) ctx[1 + k] = rng.Next(VOCAB);
        var d = ctx.AsSpan(1, DATA);
        var tgt = task switch { 0 => d[0], 1 => d[DATA - 1], 2 => Max(d), _ => Min(d) };
        return (ctx, tgt);
    }
    static int Max(Span<int> d) { var m = d[0]; foreach (var x in d) if (x > m) m = x; return m; }
    static int Min(Span<int> d) { var m = d[0]; foreach (var x in d) if (x < m) m = x; return m; }
    static double Avg(AlgFormer m, int[] skills) => skills.Average(s => Acc(m, s));
    static double Acc(AlgFormer m, int task) { var ok = 0; for (var i = 0; i < EVAL; i++) { var (c, t) = Example(task, 100_000_000L + i); if (m.Predict(c) == t) ok++; } return ok / (double)EVAL; }

    // ---- elastic coupling on the weights ----
    static void PullToMean(AlgFormer[] group, double alpha) { var mean = Average(group); foreach (var m in group) Blend(m, mean, alpha); }
    static void Blend(AlgFormer m, double[] target, double alpha) { var p = GetParams(m); for (var i = 0; i < p.Length; i++) p[i] = (1 - alpha) * p[i] + alpha * target[i]; SetParams(m, p); }
    static double Divergence(AlgFormer a, AlgFormer b) { double[] pa = GetParams(a), pb = GetParams(b); double s = 0; for (var i = 0; i < pa.Length; i++) s += (pa[i] - pb[i]) * (pa[i] - pb[i]); return Math.Sqrt(s / pa.Length); }

    // ---- parameter get/set/average ----
    static byte[] Save(AlgFormer m) { using var ms = new MemoryStream(); var w = new BinaryWriter(ms); m.Save(w); w.Flush(); return ms.ToArray(); }
    static double[] GetParams(AlgFormer m) { var b = Save(m); using var r = new BinaryReader(new MemoryStream(b)); for (var i = 0; i < 4; i++) r.ReadInt32(); var l = new List<double>(); while (r.BaseStream.Position < b.Length) l.Add(r.ReadDouble()); return l.ToArray(); }
    static void SetParams(AlgFormer m, double[] p) { var hdr = Save(m); using var ms = new MemoryStream(); ms.Write(hdr, 0, 16); var w = new BinaryWriter(ms); foreach (var d in p) w.Write(d); w.Flush(); ms.Position = 0; using var r = new BinaryReader(ms); m.Load(r); }
    static double[] Average(AlgFormer[] cells) { var ps = cells.Select(GetParams).ToArray(); var n = ps[0].Length; var avg = new double[n]; foreach (var p in ps) for (var i = 0; i < n; i++) avg[i] += p[i]; for (var i = 0; i < n; i++) avg[i] /= cells.Length; return avg; }
}
