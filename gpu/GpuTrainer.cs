// Copyright (c) 2026 Dongyang Stephen Chen, trading as Evaluated Applications. All rights reserved.
// Source-available for study and verification; commercial use, redistribution, or derivatives require permission. See LICENSE.

using PrismFormer;

namespace PrismFormer.Gpu;

/// <summary>
/// Drop-in GPU-accelerated batch trainer with a CPU/GPU LENGTH SPLIT. Short sequences (≤ <see cref="GpuMaxLen"/>) run
/// on the GPU (length-bucketed so a lone long window never pads the shorts); long sequences (which OOM the card and
/// waste the most on padding) run on the CPU in PARALLEL — so both the GPU and all cores are busy at once. Their
/// gradients sum to the identical full-batch gradient; one CPU Adam step at the end. The CPU model stays the source of
/// truth (serving/bleeding/saving read it). fp32 on the GPU half → float-close to the CPU double path.
/// </summary>
public sealed class GpuTrainer : IDisposable
{
    const int GpuMaxLen = 512;      // sequences longer than this go to the CPU (≈ half of Context=1024) — avoids GPU OOM
    const int TokenBudget = 32768;  // padded token-positions per GPU sub-batch (bounds memory + padding waste)

    readonly AlgFormer _cpu;
    readonly GpuModel _gpu;

    public GpuTrainer(AlgFormer cpu) { _cpu = cpu; _gpu = new GpuModel(cpu.Serialize()); }

    static byte[] GradBytes(float[] g) { using var ms = new MemoryStream(); var w = new BinaryWriter(ms); foreach (var x in g) w.Write((double)x); w.Flush(); return ms.ToArray(); }

    /// <summary>Train one batch, update the CPU model in place, return the mean cross-entropy loss.</summary>
    public double TrainBatch(IReadOnlyList<(int[] Ctx, int Target)> batch, double lr)
    {
        if (batch.Count == 0) return 0;
        _gpu.UpdateParams(_cpu.Serialize());   // pick up any external change (bleed/absorb) before computing grads

        var shortEx = new List<(int[] Ctx, int Target)>();
        var longEx = new List<(int[] Ctx, int Target)>();
        foreach (var e in batch) (e.Ctx.Length <= GpuMaxLen ? shortEx : longEx).Add(e);

        // CPU (long) runs concurrently with GPU (short) — both hardware halves busy at once.
        double cpuLoss = 0; AlgFormer.Grads? cpuG = null;
        var cpuTask = longEx.Count > 0
            ? System.Threading.Tasks.Task.Run(() => { cpuG = CpuGrads(longEx, out var l); cpuLoss = l; })
            : System.Threading.Tasks.Task.CompletedTask;

        double gpuLoss = 0; float[]? gpuGrads = null;
        if (shortEx.Count > 0) (gpuGrads, gpuLoss) = GpuGrads(shortEx);

        cpuTask.Wait();

        // merge: full-batch grad = GPU-half grad + CPU-half grad; one Adam step
        var merged = gpuGrads != null ? _cpu.DeserializeGradient(GradBytes(gpuGrads)) : _cpu.NewGrads();
        if (cpuG != null) merged.Add(cpuG);
        _cpu.Step(merged, lr, scale: batch.Count);
        return (gpuLoss + cpuLoss) / batch.Count;
    }

    // GPU half: length-bucketed backward over the short sequences → summed grads (float, Pairs order) + summed loss.
    (float[] grads, double lossSum) GpuGrads(List<(int[] Ctx, int Target)> ex)
    {
        var order = Enumerable.Range(0, ex.Count).OrderBy(i => ex[i].Ctx.Length).ToArray();
        float[]? sum = null; double lossSum = 0;
        for (var s = 0; s < order.Length;)
        {
            var e = s; var maxLen = 0;
            while (e < order.Length) { var len = ex[order[e]].Ctx.Length; var nm = Math.Max(maxLen, len); if (e > s && (long)(e - s + 1) * nm > TokenBudget) break; maxLen = nm; e++; }
            var n = e - s; var toks = new int[n][]; var tgts = new int[n];
            for (var i = 0; i < n; i++) { var idx = order[s + i]; toks[i] = ex[idx].Ctx; tgts[i] = ex[idx].Target; }
            var (g, loss) = _gpu.Backward(toks, tgts);
            if (sum == null) sum = g; else for (var i = 0; i < g.Length; i++) sum[i] += g[i];
            lossSum += loss * n; s = e;
        }
        return (sum!, lossSum);
    }

    // CPU half: parallel gradient accumulation over the long sequences (per-thread Grads merged) — no Step here.
    AlgFormer.Grads CpuGrads(List<(int[] Ctx, int Target)> ex, out double lossSum)
    {
        var P = Math.Max(1, Environment.ProcessorCount - 1);
        var parts = new AlgFormer.Grads[P]; var losses = new double[P];
        System.Threading.Tasks.Parallel.For(0, P, p =>
        {
            var g = _cpu.NewGrads(); var sc = _cpu.NewScratch(); double ls = 0;
            for (var i = p; i < ex.Count; i += P) ls += _cpu.Accumulate(ex[i].Ctx, ex[i].Target, g, sc);
            parts[p] = g; losses[p] = ls;
        });
        var merged = parts[0]; for (var p = 1; p < P; p++) merged.Add(parts[p]);
        lossSum = losses.Sum();
        return merged;
    }

    public void Dispose() => _gpu.Dispose();
}
