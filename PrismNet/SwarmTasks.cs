using PrismFormer;

// ═══════════════════════════════════════════════════════════════════════════════════════════════════════════
//  SwarmTasks — do skills BLEED OVER when each cell learns a different task and the colony merges? (see SWARM.md)
//
//  K cells, K tasks. The tasks are variants of ONE general skill — "copy the data token at the position named by
//  the marker": example = [marker, d0, d1, d2, d3], target = d[marker]. Cell k only ever sees marker=k, so on its
//  own it learns a single slice ("copy position k"). The GENERAL skill (indexed copy) is what emerges only if the
//  slices compose. The marker makes the tasks context-identifiable, so one shared model CAN in principle hold all
//  of them — the question is whether merging gets it there.
//
//  Three arms, so the result is honest rather than rigged:
//    ISOLATED  — each cell trained alone, never merged. Cross-task accuracy is the CONTROL (expect ~chance):
//                skills do NOT bleed without a merge channel.
//    TIGHT     — gradient-merge every round (the bit-exact colony). Expected to fuse specialists into a generalist.
//    LOOSE     — FedAvg: H local steps per cell, then average weights (the realistic swarm sync). Genuinely open.
//
//  Runnable as `prismnet swarmtasks [tasks]`.
// ═══════════════════════════════════════════════════════════════════════════════════════════════════════════

static class SwarmTasks
{
    const int VOCAB = 32, DATA = 4, MAXCTX = 1 + DATA, DMODEL = 48, SHIFTS = 12, LAYERS = 3, SEED = 1;
    const int ROUNDS = 700, BATCH = 64, H = 20, EVAL = 512;
    const double LR = 6e-3;

    static AlgFormer NewModel() => new(VOCAB, shifts: SHIFTS, layers: LAYERS, maxContext: MAXCTX, dModel: DMODEL, frozenPrefix: 0, embedSeed: null, seed: SEED);

    static readonly string[] TaskName = { "copy0", "last", "max", "min" };

    // example for (task, index): marker = task, data = 4 random tokens, target = a DIFFERENT function family per task.
    static (int[] Ctx, int Tgt) Example(int task, long index)
    {
        var rng = new Random(unchecked(1013 * task + (int)index * 7919 + 17));
        var ctx = new int[MAXCTX]; ctx[0] = task;
        for (var k = 0; k < DATA; k++) ctx[1 + k] = rng.Next(VOCAB);
        var d = ctx.AsSpan(1, DATA);
        var tgt = task switch
        {
            0 => d[0],                                   // copy first
            1 => d[DATA - 1],                            // copy last
            2 => Max(d),                                 // maximum
            _ => Min(d),                                 // minimum
        };
        return (ctx, tgt);
    }
    static int Max(Span<int> d) { var m = d[0]; foreach (var x in d) if (x > m) m = x; return m; }
    static int Min(Span<int> d) { var m = d[0]; foreach (var x in d) if (x < m) m = x; return m; }

    public static void Run(int tasks)
    {
        tasks = Math.Clamp(tasks, 2, DATA);
        Console.WriteLine("=== do skills bleed over? K cells, K tasks, one colony (see SWARM.md) ===\n");
        Console.WriteLine($"tasks (marker in ctx[0]): {string.Join(", ", TaskName.Take(tasks))}  |  model d {DMODEL}, S {SHIFTS}, L {LAYERS}  |  chance = {100.0 / VOCAB:F1}%\n");

        var trainHi = 4_000_000L;   // training draws indices [0,trainHi); eval uses a disjoint held-out range

        // ---------- ISOLATED: each cell trained alone (control) ----------
        var isolated = new AlgFormer[tasks];
        for (var k = 0; k < tasks; k++)
        {
            var m = NewModel();
            for (var r = 0; r < ROUNDS; r++) TrainBatch(m, k, r, trainHi, LR);
            isolated[k] = m;
        }
        Console.WriteLine("ISOLATED — cross-task accuracy matrix (row = model trained on task r, col = tested on task c):");
        PrintMatrix(isolated, tasks);
        Console.WriteLine("  diagonal = own task (learned); off-diagonal ~chance => a lone specialist does NOT bleed.\n");

        // ---------- TIGHT colony: gradient-merge every round (bit-exact mode) ----------
        var tight = NewModel();
        for (var r = 0; r < ROUNDS; r++)
        {
            var merged = tight.NewGrads(); var total = 0;
            for (var k = 0; k < tasks; k++)
            {
                var g = tight.NewGrads();
                for (var i = 0; i < BATCH; i++) { var (c, t) = Example(k, Idx(k, r, i, trainHi)); tight.Accumulate(c, t, g); total++; }
                merged.Add(g);
            }
            tight.Step(merged, LR, scale: total);
        }

        // ---------- LOOSE colony: FedAvg (H local steps, then average weights) ----------
        var cells = new AlgFormer[tasks]; for (var k = 0; k < tasks; k++) cells[k] = NewModel();
        for (var r = 0; r < ROUNDS; r++)
        {
            for (var k = 0; k < tasks; k++) TrainBatch(cells[k], k, r, trainHi, LR);
            if ((r + 1) % H == 0) { var avg = Average(cells); foreach (var m in cells) SetParams(m, avg); }
        }
        var loose = NewModel(); SetParams(loose, Average(cells));

        // ---------- verdict ----------
        Console.WriteLine("SKILL ON EVERY TASK — a single shared model tested across all tasks:");
        Console.WriteLine($"  {"task",-8}{"isolated-own",-14}{"TIGHT colony",-14}{"LOOSE colony",-14}");
        double tightMin = 1, looseMin = 1;
        for (var k = 0; k < tasks; k++)
        {
            double iso = Acc(isolated[k], k), ti = Acc(tight, k), lo = Acc(loose, k);
            tightMin = Math.Min(tightMin, ti); looseMin = Math.Min(looseMin, lo);
            Console.WriteLine($"  {TaskName[k],-8}{iso,10:P0}  {ti,12:P0}  {lo,12:P0}");
        }
        Console.WriteLine();
        Console.WriteLine($"TIGHT colony knows ALL {tasks} tasks (worst-task {tightMin:P0}) : {(tightMin > 0.6 ? "YES — specialists fused into one generalist" : "partial")}");
        Console.WriteLine($"LOOSE colony (FedAvg) skills survived the average (worst {looseMin:P0}) : {(looseMin > 0.6 ? "YES" : looseMin > 0.2 ? "PARTIAL — weight-averaging degrades some skill" : "NO — averaging washed skills out")}");
        Console.WriteLine("\nBleed channel = the merge: isolated specialists stay chance off their task; merging is what spreads skill across cells.");
    }

    // ---- training ----
    static long Idx(int task, int round, int i, long hi)
    {
        var h = (ulong)(round * BATCH + i) * 2654435761UL + (ulong)task * 40503UL;
        h ^= h >> 13;
        return (long)(h % (ulong)hi);
    }
    static void TrainBatch(AlgFormer m, int task, int round, long hi, double lr)
    {
        var g = m.NewGrads();
        for (var i = 0; i < BATCH; i++) { var (c, t) = Example(task, Idx(task, round, i, hi)); m.Accumulate(c, t, g); }
        m.Step(g, lr, scale: BATCH);
    }

    // ---- evaluation on a held-out index range ----
    static double Acc(AlgFormer m, int task)
    {
        var ok = 0;
        for (var i = 0; i < EVAL; i++) { var (c, t) = Example(task, 100_000_000L + i); if (m.Predict(c) == t) ok++; }
        return ok / (double)EVAL;
    }
    static void PrintMatrix(AlgFormer[] models, int tasks)
    {
        Console.Write("        "); for (var c = 0; c < tasks; c++) Console.Write($"t{c,-7}"); Console.WriteLine();
        for (var r = 0; r < tasks; r++) { Console.Write($"  t{r}   "); for (var c = 0; c < tasks; c++) Console.Write($"{Acc(models[r], c),6:P0} "); Console.WriteLine(); }
    }

    // ---- parameter get/set/average (weights = the Save doubles after the 4-int header) ----
    static byte[] Save(AlgFormer m) { using var ms = new MemoryStream(); var w = new BinaryWriter(ms); m.Save(w); w.Flush(); return ms.ToArray(); }
    static double[] GetParams(AlgFormer m) { var b = Save(m); using var r = new BinaryReader(new MemoryStream(b)); for (var i = 0; i < 4; i++) r.ReadInt32(); var l = new List<double>(); while (r.BaseStream.Position < b.Length) l.Add(r.ReadDouble()); return l.ToArray(); }
    static void SetParams(AlgFormer m, double[] p)
    {
        var hdr = Save(m); using var ms = new MemoryStream(); ms.Write(hdr, 0, 16);
        var w = new BinaryWriter(ms); foreach (var d in p) w.Write(d); w.Flush(); ms.Position = 0;
        using var r = new BinaryReader(ms); m.Load(r);
    }
    static double[] Average(AlgFormer[] cells)
    {
        var ps = cells.Select(GetParams).ToArray(); var n = ps[0].Length; var avg = new double[n];
        foreach (var p in ps) for (var i = 0; i < n; i++) avg[i] += p[i];
        for (var i = 0; i < n; i++) avg[i] /= cells.Length; return avg;
    }
}
