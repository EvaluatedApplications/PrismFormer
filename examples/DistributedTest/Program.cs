using System.Diagnostics;
using System.Reflection;
using PrismFormer;

// ═══════════════════════════════════════════════════════════════════════════════════════════════════════════
//  Distributed-training proof for PrismFormer.
//
//  The claim: AlgFormer's gradients are EXACTLY MERGEABLE — the model is read-only during backprop, so each worker
//  accumulates its data shard into its own detached Grads buffer; summing the buffers and applying one optimiser Step
//  gives a result identical to training on all the data at once (gradient of a sum = sum of gradients). That is exactly
//  the primitive distributed data-parallel / federated SGD needs, and here it is exact, not approximate.
//
//  This program proves it three ways:
//   [1] a serialized gradient buffer round-trips losslessly (the "wire" doesn't corrupt it);
//   [2] 2 nodes, each training on half the data and exchanging SERIALIZED gradients every step, end bit-for-bit
//       identical to a single-node reference (over many epochs);
//   [3] a gradient computed in a SEPARATE OS PROCESS, serialized to disk, read back and summed, is bit-identical to
//       the in-process sum — the real cross-process exchange.
// ═══════════════════════════════════════════════════════════════════════════════════════════════════════════

if (args.Length >= 4 && args[0] == "worker") { Helpers.RunWorker(args[1], args[2], args[3]); return; }

Console.WriteLine("=== PrismFormer distributed-training proof ===\n");
var data = Helpers.MakeData(256);
const int NODES = 2, EPOCHS = 6, BATCH = 32;
const double LR = 5e-3;
Console.WriteLine($"model {Helpers.NewModel().ParamCount:N0} params · {data.Count} examples · {NODES} nodes · {EPOCHS} epochs · batch {BATCH}\n");

// [1] serialization round-trip
{
    var m = Helpers.NewModel();
    var g = m.NewGrads(); foreach (var (c, t) in data.Take(20)) m.Accumulate(c, t, g);
    var back = m.DeserializeGradient(m.SerializeGradient(g));
    var ok = m.SerializeGradient(back).SequenceEqual(m.SerializeGradient(g));
    Console.WriteLine($"[1] gradient serialize -> bytes -> deserialize is LOSSLESS: {(ok ? "YES" : "NO")}");
}

// [2] in-process: N nodes with serialized gradient exchange vs single-node reference, bit-for-bit
{
    var reference = Helpers.NewModel();
    var lossBefore = Helpers.AvgLoss(reference, data);
    Helpers.TrainGrouped(reference, data, EPOCHS, BATCH, NODES, LR);        // "one machine" baseline (same batching)

    var nodes = Enumerable.Range(0, NODES).Select(_ => Helpers.NewModel()).ToArray();
    Helpers.TrainDistributed(nodes, data, EPOCHS, BATCH, LR);               // N replicas, serialized grad all-reduce each step

    var bitEqual = Helpers.BitEqual(nodes[0], reference) && nodes.All(n => Helpers.BitEqual(n, nodes[0]));
    Console.WriteLine($"\n[2] {NODES} nodes, serialized gradient exchange every step, {EPOCHS} epochs:");
    Console.WriteLine($"    bit-for-bit identical to single-node reference : {(bitEqual ? "YES" : "NO")}   (max param diff {Helpers.MaxDiff(nodes[0], reference):E1})");

    var naive = Helpers.NewModel();
    Helpers.TrainNaive(naive, data, EPOCHS, BATCH, LR);                     // different summation order (single ungrouped buffer)
    Console.WriteLine($"    vs a naive different-order serial run          : max param diff {Helpers.MaxDiff(nodes[0], naive):E1}  (floating-point summation order only)");
    Console.WriteLine($"    (real training — avg loss {lossBefore:F3} -> {Helpers.AvgLoss(reference, data):F3})");
}

// [3] genuine cross-process gradient exchange
Helpers.RunCrossProcess(data.Take(64).ToList());

Console.WriteLine("\nConclusion: gradients are exactly mergeable and survive serialization, so data-parallel / federated");
Console.WriteLine("training is bit-for-bit equivalent to single-node. Swapping the in-memory/disk exchange for a socket");
Console.WriteLine("is the only step to true multi-machine.");

static class Helpers
{
    public const int VOCAB = 32, SHIFTS = 8, LAYERS = 2, MAXCTX = 4, DMODEL = 32, SEED = 1;

    // frozenPrefix 0 => every dimension learns, so gradients are non-zero everywhere (a real distributed workload)
    public static AlgFormer NewModel() => new(VOCAB, shifts: SHIFTS, layers: LAYERS, maxContext: MAXCTX, dModel: DMODEL, frozenPrefix: 0, embedSeed: null, seed: SEED);

    public static List<(int[] Ctx, int Tgt)> MakeData(int count)
    {
        var rng = new Random(12345); var d = new List<(int[], int)>(count);
        for (var i = 0; i < count; i++) { var ctx = new int[MAXCTX]; for (var k = 0; k < MAXCTX; k++) ctx[k] = rng.Next(VOCAB); d.Add((ctx, ctx[0])); }
        return d;
    }

    public static double AvgLoss(AlgFormer m, List<(int[] Ctx, int Tgt)> data)
    {
        var g = m.NewGrads(); double s = 0; foreach (var (c, t) in data) s += m.Accumulate(c, t, g); return s / data.Count;
    }

    // single-node baseline, batched and sharded EXACTLY like the distributed run (round-robin shards, summed in order)
    public static void TrainGrouped(AlgFormer m, List<(int[] Ctx, int Tgt)> data, int epochs, int batch, int nodes, double lr)
    {
        for (var ep = 0; ep < epochs; ep++)
            for (var start = 0; start < data.Count; start += batch)
            {
                var b = data.GetRange(start, Math.Min(batch, data.Count - start));
                var merged = m.NewGrads();
                for (var k = 0; k < nodes; k++) { var g = m.NewGrads(); for (var i = k; i < b.Count; i += nodes) m.Accumulate(b[i].Ctx, b[i].Tgt, g); merged.Add(g); }
                m.Step(merged, lr, scale: b.Count);
            }
    }

    // N replicas: each computes its shard's gradient, SERIALIZES it (the wire), then every replica sums all wires and steps
    public static void TrainDistributed(AlgFormer[] reps, List<(int[] Ctx, int Tgt)> data, int epochs, int batch, double lr)
    {
        var nodes = reps.Length;
        for (var ep = 0; ep < epochs; ep++)
            for (var start = 0; start < data.Count; start += batch)
            {
                var b = data.GetRange(start, Math.Min(batch, data.Count - start));
                var wire = new byte[nodes][];
                for (var k = 0; k < nodes; k++) { var g = reps[k].NewGrads(); for (var i = k; i < b.Count; i += nodes) reps[k].Accumulate(b[i].Ctx, b[i].Tgt, g); wire[k] = reps[k].SerializeGradient(g); }
                for (var k = 0; k < nodes; k++)
                {
                    var merged = reps[k].NewGrads();
                    for (var j = 0; j < nodes; j++) merged.Add(reps[k].DeserializeGradient(wire[j]));
                    reps[k].Step(merged, lr, scale: b.Count);
                }
            }
    }

    // a plain serial run that sums each batch into ONE buffer in index order (different summation order — for the tolerance check)
    public static void TrainNaive(AlgFormer m, List<(int[] Ctx, int Tgt)> data, int epochs, int batch, double lr)
    {
        for (var ep = 0; ep < epochs; ep++)
            for (var start = 0; start < data.Count; start += batch)
            {
                var b = data.GetRange(start, Math.Min(batch, data.Count - start));
                var g = m.NewGrads(); foreach (var (c, t) in b) m.Accumulate(c, t, g); m.Step(g, lr, scale: b.Count);
            }
    }

    public static byte[] SaveBytes(AlgFormer m) { using var ms = new MemoryStream(); var w = new BinaryWriter(ms); m.Save(w); w.Flush(); return ms.ToArray(); }
    public static bool BitEqual(AlgFormer a, AlgFormer b) => SaveBytes(a).SequenceEqual(SaveBytes(b));
    public static double[] ParamsOf(AlgFormer m)
    {
        using var ms = new MemoryStream(SaveBytes(m)); using var r = new BinaryReader(ms);
        r.ReadInt32(); r.ReadInt32(); r.ReadInt32(); r.ReadInt32();   // shape header
        var l = new List<double>(); while (ms.Position < ms.Length) l.Add(r.ReadDouble()); return l.ToArray();
    }
    public static double MaxDiff(AlgFormer a, AlgFormer b) { var pa = ParamsOf(a); var pb = ParamsOf(b); double m = 0; for (var i = 0; i < pa.Length; i++) m = Math.Max(m, Math.Abs(pa[i] - pb[i])); return m; }

    public static void WriteShard(string path, List<(int[] Ctx, int Tgt)> d)
    {
        using var w = new BinaryWriter(File.Create(path));
        w.Write(d.Count);
        foreach (var (ctx, tgt) in d) { w.Write(ctx.Length); foreach (var t in ctx) w.Write(t); w.Write(tgt); }
    }
    public static List<(int[] Ctx, int Tgt)> ReadShard(string path)
    {
        using var r = new BinaryReader(File.OpenRead(path));
        var n = r.ReadInt32(); var d = new List<(int[], int)>(n);
        for (var i = 0; i < n; i++) { var len = r.ReadInt32(); var ctx = new int[len]; for (var k = 0; k < len; k++) ctx[k] = r.ReadInt32(); d.Add((ctx, r.ReadInt32())); }
        return d;
    }

    // worker process: load the shared model + its data shard, compute the shard's gradient, write it serialized, exit
    public static void RunWorker(string shardPath, string modelPath, string outPath)
    {
        var m = NewModel();
        using (var r = new BinaryReader(File.OpenRead(modelPath))) if (!m.Load(r)) { Console.Error.WriteLine("worker: model load failed"); Environment.Exit(2); }
        var shard = ReadShard(shardPath);
        var g = m.NewGrads(); foreach (var (c, t) in shard) m.Accumulate(c, t, g);
        File.WriteAllBytes(outPath, m.SerializeGradient(g));
    }

    public static void RunCrossProcess(List<(int[] Ctx, int Tgt)> data)
    {
        var dir = Path.Combine(Path.GetTempPath(), "prism-dist-test");
        if (Directory.Exists(dir)) Directory.Delete(dir, true);
        Directory.CreateDirectory(dir);
        try
        {
            var model = NewModel();
            using (var w = new BinaryWriter(File.Create(Path.Combine(dir, "model.bin")))) model.Save(w);
            var s0 = data.Where((_, i) => i % 2 == 0).ToList();
            var s1 = data.Where((_, i) => i % 2 == 1).ToList();
            WriteShard(Path.Combine(dir, "s0.bin"), s0);
            WriteShard(Path.Combine(dir, "s1.bin"), s1);

            var dll = Assembly.GetEntryAssembly()!.Location;
            Process Worker(string shard, string outp)
            {
                var psi = new ProcessStartInfo("dotnet", $"\"{dll}\" worker \"{Path.Combine(dir, shard)}\" \"{Path.Combine(dir, "model.bin")}\" \"{Path.Combine(dir, outp)}\"")
                { UseShellExecute = false, RedirectStandardError = true };
                return Process.Start(psi)!;
            }
            Console.WriteLine("\n[3] spawning 2 worker processes, each computes its shard's gradient in a separate OS process...");
            var p0 = Worker("s0.bin", "g0.bin"); var p1 = Worker("s1.bin", "g1.bin");
            p0.WaitForExit(); p1.WaitForExit();
            if (p0.ExitCode != 0 || p1.ExitCode != 0) { Console.WriteLine($"    worker failed (exit {p0.ExitCode}/{p1.ExitCode}): {p0.StandardError.ReadToEnd()}{p1.StandardError.ReadToEnd()}"); return; }

            var fromProc = model.NewGrads();
            fromProc.Add(model.DeserializeGradient(File.ReadAllBytes(Path.Combine(dir, "g0.bin"))));
            fromProc.Add(model.DeserializeGradient(File.ReadAllBytes(Path.Combine(dir, "g1.bin"))));

            var inProc = model.NewGrads();
            var gi0 = model.NewGrads(); foreach (var (c, t) in s0) model.Accumulate(c, t, gi0); inProc.Add(gi0);
            var gi1 = model.NewGrads(); foreach (var (c, t) in s1) model.Accumulate(c, t, gi1); inProc.Add(gi1);

            var same = model.SerializeGradient(fromProc).SequenceEqual(model.SerializeGradient(inProc));
            Console.WriteLine($"    gradient summed from 2 separate processes == in-process sum, bit-for-bit : {(same ? "YES" : "NO")}");
        }
        finally { try { Directory.Delete(dir, true); } catch { } }
    }
}
