// Copyright (c) 2026 Dongyang Stephen Chen, trading as Evaluated Applications. All rights reserved.
// Source-available for study and verification; commercial use, redistribution, or derivatives require permission. See LICENSE.

using System.Diagnostics;
using PrismFormer;

namespace PrismFormer.Gpu;

/// <summary>
/// TRAINING-FIT probe: how BIG a model can we train on this GPU (fp32, forward+backward via <see cref="GpuTrainer"/>)?
/// Builds ONE config, runs a few real train batches at a target context, and reports params + ms/batch + tokens/s, or
/// FAIL (out-of-memory / kernel limit). Run one config per process (a ladder driven from the shell) so an OOM on a big
/// rung only kills that process, not the whole sweep. This finds the ceiling to MAX the hardware for the reset model.
/// </summary>
public static class GpuFit
{
    public static void Run(int d, int L, int c, int tok, int V = 4096, int S = 64, int bs = 128)
    {
        var cpu = new AlgFormer(V, shifts: S, layers: L, maxContext: c, dModel: d, frozenPrefix: Math.Min(PhasorCodec.FrozenReals, d), embedSeed: null, seed: 1);
        Console.WriteLine($"d{d,-4} L{L,-3} c{c,-4} S{S} V{V} tok{tok,-6} · {cpu.ParamCount / 1e6,5:F1}M params · {GpuDevice.Describe}");
        try
        {
            using var gt = new GpuTrainer(cpu, tokenBudget: tok);
            var rng = new Random(1);
            List<(int[], int)> B() { var l = new List<(int[], int)>(bs); for (var i = 0; i < bs; i++) { var x = new int[c]; for (var t = 0; t < c; t++) x[t] = rng.Next(V); l.Add((x, rng.Next(V))); } return l; }
            gt.TrainBatch(B(), 1e-3);   // warm up (JIT + buffer alloc — this is where OOM shows)
            var sw = Stopwatch.StartNew();
            const int reps = 4; for (var r = 0; r < reps; r++) gt.TrainBatch(B(), 1e-3);
            var ms = sw.Elapsed.TotalMilliseconds / reps;
            Console.WriteLine($"   OK    {ms,7:F0} ms/batch (bs{bs} ctx{c})   {bs * c * 1000.0 / ms,8:F0} tok/s");
        }
        catch (Exception e) { Console.WriteLine($"   FAIL  {e.GetType().Name}: {e.Message.Split('\n')[0].Trim()}"); }
        GpuDevice.Shutdown();
    }
}
