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

    /// <summary>Train one batch on the GPU, update the CPU model in place, return the mean cross-entropy loss.</summary>
    public double TrainBatch(IReadOnlyList<(int[] Ctx, int Target)> batch, double lr)
    {
        _gpu.UpdateParams(_cpu.Serialize());   // pick up any external change to the CPU model (bleed/absorb) before computing grads
        var toks = new int[batch.Count][]; var tgts = new int[batch.Count];
        for (var i = 0; i < batch.Count; i++) { toks[i] = batch[i].Ctx; tgts[i] = batch[i].Target; }
        var (grads, loss) = _gpu.Backward(toks, tgts);
        var g = _cpu.DeserializeGradient(GradBytes(grads));
        _cpu.Step(g, lr, scale: batch.Count);   // CPU Adam owns the moment state + frozen-codec pin
        return loss;
    }

    public void Dispose() => _gpu.Dispose();
}
