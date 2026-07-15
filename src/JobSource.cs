// Copyright (c) 2026 Dongyang Stephen Chen, trading as Evaluated Applications. All rights reserved.
// Source-available for study and verification; commercial use, redistribution, or derivatives require permission. See LICENSE.

namespace PrismFormer;

/// <summary>
/// A deterministic, index-addressable training-data source. Every node builds the SAME source from a manifest and
/// materializes example <c>i</c> byte-identically — so a distributed host ships only the manifest (once), the model
/// (once), example POSITIONS ("compute i..j", a few bytes), and gradients. The data itself is never sent: each worker
/// reconstructs it locally from the same source. BabyLM and Gym both reduce to this (they generate data from code).
/// </summary>
public interface IJobSource
{
    long Count { get; }
    (int[] Ctx, int Target) GetExample(long index);
}

/// <summary>Trains the model on a set of example POSITIONS (indices into an <see cref="IJobSource"/>). Implemented by a
/// local trainer (reconstruct + train on cores) and by the relay host (delegate positions to slaves).</summary>
public interface IJobTrainer
{
    double TrainPositions(IReadOnlyList<long> positions, double lr);
}

/// <summary>Local trainer: reconstruct the assigned examples from the source and train on local cores.</summary>
public sealed class LocalJobTrainer : IJobTrainer
{
    readonly IJobSource _source;
    readonly PrismTrainer _trainer;
    public LocalJobTrainer(AlgFormer model, IJobSource source) { _source = source; _trainer = new PrismTrainer(model); }
    public double TrainPositions(IReadOnlyList<long> positions, double lr)
    {
        var batch = new List<(int[] Ctx, int Target)>(positions.Count);
        foreach (var p in positions) batch.Add(_source.GetExample(p));
        return _trainer.TrainBatch(batch, lr);
    }
}

/// <summary>A deterministic synthetic source (for tests/demos): example <c>i</c> is a fixed function of its index, so
/// any node reproduces it exactly. Target = first context token (a learnable pattern).</summary>
public sealed class SyntheticJobSource : IJobSource
{
    readonly int _vocab, _ctxLen, _seed; readonly long _count;
    public SyntheticJobSource(long count, int vocab = 32, int ctxLen = 4, int seed = 12345) { _count = count; _vocab = vocab; _ctxLen = ctxLen; _seed = seed; }
    public long Count => _count;
    public (int[] Ctx, int Target) GetExample(long index)
    {
        var rng = new Random(_seed + (int)index);
        var ctx = new int[_ctxLen]; for (var k = 0; k < _ctxLen; k++) ctx[k] = rng.Next(_vocab);
        return (ctx, ctx[0]);
    }
    public byte[] Manifest() { using var ms = new MemoryStream(); var w = new BinaryWriter(ms); w.Write((byte)0); w.Write(_count); w.Write(_vocab); w.Write(_ctxLen); w.Write(_seed); w.Flush(); return ms.ToArray(); }
    public static SyntheticJobSource FromManifest(byte[] m) { using var r = new BinaryReader(new MemoryStream(m)); r.ReadByte(); var count = r.ReadInt64(); var vocab = r.ReadInt32(); var ctxLen = r.ReadInt32(); var seed = r.ReadInt32(); return new SyntheticJobSource(count, vocab, ctxLen, seed); }
}
