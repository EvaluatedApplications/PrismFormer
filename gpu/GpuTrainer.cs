// Copyright (c) 2026 Dongyang Stephen Chen, trading as Evaluated Applications. All rights reserved.
// Source-available for study and verification; commercial use, redistribution, or derivatives require permission. See LICENSE.

using PrismFormer;

namespace PrismFormer.Gpu;

/// <summary>
/// Drop-in GPU-accelerated batch trainer for an AlgFormer. The CPU model stays the SOURCE OF TRUTH (serving, bleeding,
/// saving all read it); the GPU only accelerates forward+backward. Each batch: sync GPU params from the CPU model
/// (so external changes — weight-slice bleed, absorb — are picked up), run GPU fwd+bwd, apply CPU Adam via
/// AlgFormer.Step (which owns the optimiser state and the frozen-codec pin). fp32 on GPU → float-close to the CPU
/// double path. Only worth using when GpuDevice.HasGpu AND the batch is big enough that GPU compute beats the
/// per-batch param-sync overhead (measure with `prismformer-gpu studiobench`).
/// </summary>
public sealed class GpuTrainer : IDisposable
{
    readonly AlgFormer _cpu;
    readonly GpuModel _gpu;

    public GpuTrainer(AlgFormer cpu) { _cpu = cpu; _gpu = new GpuModel(cpu.Serialize()); }

    static byte[] GradBytes(float[] g) { using var ms = new MemoryStream(); var w = new BinaryWriter(ms); foreach (var x in g) w.Write((double)x); w.Flush(); return ms.ToArray(); }

    // Cap on padded token-positions per GPU sub-batch (bounds memory AND padding waste on ragged batches).
    const int TokenBudget = 32768;

    /// <summary>Train one batch on the GPU, update the CPU model in place, return the mean cross-entropy loss.
    /// LENGTH-BUCKETED: examples are sorted by length and run in same-length sub-batches, so a lone long window never
    /// pads the short ones (right-pad-to-max would otherwise waste ~10x on ragged batches). Grads sum to the identical
    /// result as one big batch; one CPU Adam step at the end.</summary>
    public double TrainBatch(IReadOnlyList<(int[] Ctx, int Target)> batch, double lr)
    {
        if (batch.Count == 0) return 0;
        _gpu.UpdateParams(_cpu.Serialize());   // pick up any external change (bleed/absorb) before computing grads
        var order = Enumerable.Range(0, batch.Count).OrderBy(i => batch[i].Ctx.Length).ToArray();
        float[]? gradSum = null; double lossSum = 0;
        for (var s = 0; s < order.Length;)
        {
            var e = s; var maxLen = 0;                           // grow the sub-batch until (count × maxLen) would exceed the budget
            while (e < order.Length) { var len = batch[order[e]].Ctx.Length; var nm = Math.Max(maxLen, len); if (e > s && (long)(e - s + 1) * nm > TokenBudget) break; maxLen = nm; e++; }
            var n = e - s; var toks = new int[n][]; var tgts = new int[n];
            for (var i = 0; i < n; i++) { var idx = order[s + i]; toks[i] = batch[idx].Ctx; tgts[i] = batch[idx].Target; }
            var (g, loss) = _gpu.Backward(toks, tgts);
            if (gradSum == null) gradSum = g; else for (var i = 0; i < g.Length; i++) gradSum[i] += g[i];
            lossSum += loss * n; s = e;
        }
        var gg = _cpu.DeserializeGradient(GradBytes(gradSum!));
        _cpu.Step(gg, lr, scale: batch.Count);   // sum of sub-batch grads == full-batch grad; one Adam step
        return lossSum / batch.Count;
    }

    public void Dispose() => _gpu.Dispose();
}
