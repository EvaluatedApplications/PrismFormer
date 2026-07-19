// Copyright (c) 2026 Dongyang Stephen Chen, trading as Evaluated Applications. All rights reserved.
// Source-available for study and verification; commercial use, redistribution, or derivatives require permission. See LICENSE.

using PrismFormer;

namespace PrismFormer.Bench;

/// <summary>
/// SWARM DEMONSTRATION (--swarmdemo) — a clean, self-contained bench that DEMONSTRATES (not asserts) the two swarm
/// mechanisms a reviewer flagged, using a handful of SEPARATELY-instantiated models and a before/after table for each.
///
///   PART 1 — MASTER–SLAVE GRADIENT SHARING (bit-exact mergeable gradients).
///     A master + K slave models, each slave holding ONE skill's data shard. Every round each slave computes the
///     gradient for its own skill on its current weights and ships it (SerializeGradient); the master merges them
///     (Grads.Add) and applies one Step; the merged step is applied back to every slave (so a slave ACQUIRES skills
///     it never computed a gradient for). Two things are shown:
///       (a) a slave that only ever computed the copy0 gradient ends up competent on ALL K skills (shared gradients);
///       (b) BIT-EXACTNESS: the master's final weights equal a single-process model trained on the UNION of the
///           shards, to the last bit (max abs param diff = 0) — the merge is a lossless gradient sum, not an average.
///
///   PART 2 — AVERAGING → SKILL BLEED (elastic weight-averaging, paper §4.8).
///     Two separately-trained models sharing one init: A trains ONLY skill A, B trains ONLY skill B (each high on its
///     own, ~chance on the other). Then they are elastically weight-averaged (pull-to-mean) over some rounds while
///     each keeps training its own skill. The other node's skill BLEEDS across the average: both end up competent on
///     both skills. Printed as a model x skill accuracy matrix, before and after.
///
///   Toy skills (reused from the swarm tests): ctx = [marker, d0..d3], target = a per-marker function —
///     copy0 = d0, last = d3, max = max(d), min = min(d). Production toy dims (d48, S12, L3); cost bounded by rounds.
///   Usage: prismformer-bench --swarmdemo [--rounds N]   (N scales the round budget for a fast smoke run).
/// </summary>
internal static class SwarmDemoBench
{
    // Production toy-swarm dims (identical to SwarmTasks/SwarmBleed in PrismNet).
    const int VOCAB = 32, DATA = 4, MAXCTX = 1 + DATA, DMODEL = 48, SHIFTS = 12, LAYERS = 3, SEED = 1;
    const int BATCH = 64, EVAL = 512;
    const double LR = 6e-3, CHANCE = 100.0 / VOCAB;
    static readonly string[] TaskName = { "copy0", "last", "max", "min" };

    static AlgFormer NewModel() => new(VOCAB, shifts: SHIFTS, layers: LAYERS, maxContext: MAXCTX, dModel: DMODEL, frozenPrefix: 0, embedSeed: null, seed: SEED);

    public static void Run(string[] args)
    {
        try { Console.OutputEncoding = System.Text.Encoding.UTF8; } catch { }
        int? over = null;
        for (var i = 0; i < args.Length - 1; i++) if (args[i] == "--rounds") over = int.Parse(args[i + 1]);

        int p1Rounds = over ?? 400;           // Part 1 merge rounds
        int warmRounds = over ?? 300;          // Part 2 isolated warmup
        int coupleRounds = over.HasValue ? over.Value * 2 : 900;   // Part 2 elastic-averaging rounds

        Console.WriteLine("=== SWARM DEMONSTRATION — separately-trained models, concrete before/after (see SWARM.md) ===");
        Console.WriteLine($"toy skills (marker in ctx[0]): {string.Join(", ", TaskName)}   |   dims d{DMODEL} S{SHIFTS} L{LAYERS}   |   chance {CHANCE:F1}%");
        Console.WriteLine($"budget: part1 {p1Rounds} merge rounds, part2 {warmRounds} warmup + {coupleRounds} averaging rounds (x{BATCH}/skill)\n");

        Part1_MasterSlave(p1Rounds);
        Console.WriteLine();
        Part2_AveragingBleed(warmRounds, coupleRounds);
    }

    // ─────────────────────────────────────────────────────────────────────────────────────────────────────────
    //  PART 1 — master + K slaves; sharded gradient sum == single-process union training, to the bit.
    // ─────────────────────────────────────────────────────────────────────────────────────────────────────────
    static void Part1_MasterSlave(int rounds)
    {
        const int K = 4;                         // one slave per skill
        Console.WriteLine($"PART 1 — MASTER–SLAVE GRADIENT SHARING ({K} slaves, slave k holds ONLY skill k's data shard)");
        Console.WriteLine($"  each round: every slave computes its skill's gradient, master merges (Grads.Add) + Steps, merged step applied back.\n");

        var master = NewModel();
        var slaves = new AlgFormer[K]; for (var k = 0; k < K; k++) slaves[k] = NewModel();   // identical init to master
        var reference = NewModel();              // single-process control: trained on the UNION of all shards
        var isoSlave = NewModel();               // control: trains ONLY skill 0, never joins the swarm

        var before = Enumerable.Range(0, K).Select(k => Acc(master, k)).ToArray();

        for (var r = 0; r < rounds; r++)
        {
            // ---- swarm: each slave ships its skill's gradient; master merges + steps; step re-applied to every slave ----
            var merged = master.NewGrads(); var total = 0;
            for (var k = 0; k < K; k++)
            {
                var g = slaves[k].NewGrads();
                for (var i = 0; i < BATCH; i++) { var (c, t) = Example(k, Idx(k, r, i)); slaves[k].Accumulate(c, t, g); total++; }
                merged.Add(master.DeserializeGradient(slaves[k].SerializeGradient(g)));   // ship over the (in-proc) wire, round-trip
            }
            master.Step(merged, LR, scale: total);
            for (var k = 0; k < K; k++) slaves[k].Step(merged, LR, scale: total);         // APPLY: slaves acquire the merged update

            // ---- single-process reference: identical per-skill order, merged the same way, one model ----
            var refMerged = reference.NewGrads(); var refTotal = 0;
            for (var k = 0; k < K; k++)
            {
                var g = reference.NewGrads();
                for (var i = 0; i < BATCH; i++) { var (c, t) = Example(k, Idx(k, r, i)); reference.Accumulate(c, t, g); refTotal++; }
                refMerged.Add(g);
            }
            reference.Step(refMerged, LR, scale: refTotal);

            // ---- isolated slave: only ever sees skill 0 (control for "did the swarm actually teach the others?") ----
            var ig = isoSlave.NewGrads();
            for (var i = 0; i < BATCH; i++) { var (c, t) = Example(0, Idx(0, r, i)); isoSlave.Accumulate(c, t, ig); }
            isoSlave.Step(ig, LR, scale: BATCH);
        }

        // ---- (a) skill acquisition: master + the copy0-only slave now know every skill; the isolated copy0 model does not
        var after = Enumerable.Range(0, K).Select(k => Acc(master, k)).ToArray();
        Console.WriteLine($"  (a) SKILL ACQUISITION — accuracy per skill:");
        Console.Write($"      {"model",-30}"); foreach (var n in TaskName) Console.Write($"{n,8}"); Console.WriteLine();
        Console.Write($"      {"master (fresh, before)",-30}"); foreach (var b in before) Console.Write($"{b,8:P0}"); Console.WriteLine();
        Console.Write($"      {"master (after merged rounds)",-30}"); foreach (var a in after) Console.Write($"{a,8:P0}"); Console.WriteLine();
        Console.Write($"      {"swarm slave0 (copy0 grads only)",-30}"); for (var k = 0; k < K; k++) Console.Write($"{Acc(slaves[0], k),8:P0}"); Console.WriteLine();
        Console.Write($"      {"ISOLATED copy0-only (control)",-30}"); for (var k = 0; k < K; k++) Console.Write($"{Acc(isoSlave, k),8:P0}"); Console.WriteLine();
        Console.WriteLine("      slave0 never computed a gradient for last/max/min, yet knows them — acquired purely from the shared merged step.");
        Console.WriteLine("      the isolated copy0 model stays ~chance off its one skill: the swarm merge is what spread the skills.\n");

        // ---- (b) bit-exactness: master (sharded sum) vs reference (single-process union) ----
        var diff = MaxDiff(master, reference);
        var equal = ByteEqual(master, reference);
        Console.WriteLine($"  (b) BIT-EXACT MERGE — master (K sharded gradients summed) vs single-process training on the union:");
        Console.WriteLine($"      max abs param diff = {diff:E1}   byte-identical = {equal}   =>  {(equal && diff == 0.0 ? "BIT-EXACT (lossless gradient sum)" : "NOT bit-exact")}");
    }

    // ─────────────────────────────────────────────────────────────────────────────────────────────────────────
    //  PART 2 — two separately-trained specialists; elastic weight-averaging bleeds each skill into the other.
    // ─────────────────────────────────────────────────────────────────────────────────────────────────────────
    static void Part2_AveragingBleed(int warmRounds, int coupleRounds)
    {
        const int skillA = 0, skillB = 2;        // copy0 vs max — clearly separable
        const int H = 15; const double alpha = 0.5;
        Console.WriteLine($"PART 2 — AVERAGING → SKILL BLEED (model A trains ONLY {TaskName[skillA]}, model B trains ONLY {TaskName[skillB]}; shared init)");
        Console.WriteLine($"  warmup {warmRounds} rounds isolated, then elastic weight-average (pull-to-mean alpha {alpha} every {H} rounds) for {coupleRounds} rounds.\n");

        var A = NewModel(); var B = NewModel();   // shared init (same seed) => same loss basin, so averaging is meaningful

        // ---- warmup: each learns only its own skill, no coupling ----
        for (var r = 0; r < warmRounds; r++) { TrainBatch(A, skillA, r); TrainBatch(B, skillB, r); }
        var beforeMatrix = Matrix(A, B, skillA, skillB);
        var beforeDiv = Divergence(A, B);

        // ---- elastic averaging: keep training own skill, but pull both toward the mean periodically ----
        for (var r = 0; r < coupleRounds; r++)
        {
            TrainBatch(A, skillA, warmRounds + r);
            TrainBatch(B, skillB, warmRounds + r);
            if ((r + 1) % H == 0) PullToMean(new[] { A, B }, alpha);
        }
        var afterMatrix = Matrix(A, B, skillA, skillB);
        var afterDiv = Divergence(A, B);

        void PrintMatrix(string title, double[][] m, double div)
        {
            Console.WriteLine($"  {title}");
            Console.WriteLine($"      {"model",-14}{TaskName[skillA],9}{TaskName[skillB],9}");
            Console.WriteLine($"      {("A (" + TaskName[skillA] + ")"),-14}{m[0][0],9:P0}{m[0][1],9:P0}");
            Console.WriteLine($"      {("B (" + TaskName[skillB] + ")"),-14}{m[1][0],9:P0}{m[1][1],9:P0}");
            Console.WriteLine($"      divergence(A,B) = {div:F3}  ({(div < 0.02 ? "collapsed to copies" : "still distinct models")})");
        }
        PrintMatrix("BEFORE (isolated specialists):", beforeMatrix, beforeDiv);
        Console.WriteLine();
        PrintMatrix("AFTER (elastic weight-averaging):", afterMatrix, afterDiv);
        Console.WriteLine();
        var bled = afterMatrix[0][1] > 2 * CHANCE / 100 && afterMatrix[1][0] > 2 * CHANCE / 100;
        Console.WriteLine($"  A gained {TaskName[skillB]}: {beforeMatrix[0][1]:P0} -> {afterMatrix[0][1]:P0}   |   B gained {TaskName[skillA]}: {beforeMatrix[1][0]:P0} -> {afterMatrix[1][0]:P0}");
        Console.WriteLine($"  skill bled across the average (both above chance on the OTHER skill): {(bled ? "YES" : "NO")}");
    }

    static double[][] Matrix(AlgFormer a, AlgFormer b, int sA, int sB)
        => new[] { new[] { Acc(a, sA), Acc(a, sB) }, new[] { Acc(b, sA), Acc(b, sB) } };

    // ── skills (identical to SwarmTasks): ctx = [marker, d0..d3]; target is a per-marker function of the data ──
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
    static long Idx(int task, int round, int i) { var h = (ulong)(round * BATCH + i) * 2654435761UL + (ulong)task * 40503UL; h ^= h >> 13; return (long)(h % 4_000_000UL); }
    static void TrainBatch(AlgFormer m, int task, int round)
    {
        var g = m.NewGrads();
        for (var i = 0; i < BATCH; i++) { var (c, t) = Example(task, Idx(task, round, i)); m.Accumulate(c, t, g); }
        m.Step(g, LR, scale: BATCH);
    }
    static double Acc(AlgFormer m, int task) { var ok = 0; for (var i = 0; i < EVAL; i++) { var (c, t) = Example(task, 100_000_000L + i); if (m.Predict(c) == t) ok++; } return ok / (double)EVAL; }

    // ── weight get/set/average (the doubles after the 4-int Save header) ──
    static byte[] Save(AlgFormer m) { using var ms = new MemoryStream(); var w = new BinaryWriter(ms); m.Save(w); w.Flush(); return ms.ToArray(); }
    static double[] Params(AlgFormer m) { var b = Save(m); using var r = new BinaryReader(new MemoryStream(b)); for (var i = 0; i < 4; i++) r.ReadInt32(); var l = new List<double>(); while (r.BaseStream.Position < b.Length) l.Add(r.ReadDouble()); return l.ToArray(); }
    static void SetParams(AlgFormer m, double[] p) { var hdr = Save(m); using var ms = new MemoryStream(); ms.Write(hdr, 0, 16); var w = new BinaryWriter(ms); foreach (var d in p) w.Write(d); w.Flush(); ms.Position = 0; using var r = new BinaryReader(ms); m.Load(r); }
    static bool ByteEqual(AlgFormer a, AlgFormer b) => Save(a).SequenceEqual(Save(b));
    static double MaxDiff(AlgFormer a, AlgFormer b) { double[] pa = Params(a), pb = Params(b); double m = 0; for (var i = 0; i < pa.Length; i++) m = Math.Max(m, Math.Abs(pa[i] - pb[i])); return m; }
    static double Divergence(AlgFormer a, AlgFormer b) { double[] pa = Params(a), pb = Params(b); double s = 0; for (var i = 0; i < pa.Length; i++) s += (pa[i] - pb[i]) * (pa[i] - pb[i]); return Math.Sqrt(s / pa.Length); }
    static double[] Average(AlgFormer[] cells) { var ps = cells.Select(Params).ToArray(); var n = ps[0].Length; var avg = new double[n]; foreach (var p in ps) for (var i = 0; i < n; i++) avg[i] += p[i]; for (var i = 0; i < n; i++) avg[i] /= cells.Length; return avg; }
    static void PullToMean(AlgFormer[] group, double alpha) { var mean = Average(group); foreach (var m in group) { var p = Params(m); for (var i = 0; i < p.Length; i++) p[i] = (1 - alpha) * p[i] + alpha * mean[i]; SetParams(m, p); } }
}
