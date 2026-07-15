// Copyright (c) 2026 Dongyang Stephen Chen, trading as Evaluated Applications. All rights reserved.
// Source-available for study and verification; commercial use, redistribution, or derivatives require permission. See LICENSE.

using EvalApp.Consumer;

namespace PrismFormer;

/// <summary>Applies one minibatch (compute its gradient, step the model). Implemented by the local
/// <see cref="PrismTrainer"/> and the networked <see cref="DistributedHost"/>, so a training loop can target either
/// interchangeably — local cores, or local cores + networked slaves.</summary>
public interface IBatchTrainer
{
    double TrainBatch(IReadOnlyList<(int[] Ctx, int Target)> batch, double lr, CancellationToken ct = default);
}

/// <summary>
/// Data-parallel trainer for <see cref="AlgFormer"/>. A minibatch is split into shards; each shard accumulates
/// gradients into its OWN <see cref="AlgFormer.Grads"/> buffer (the model is read-only during backprop, so shards are
/// race-free); the shard grads are summed and applied with one Adam <see cref="AlgFormer.Step"/>. Because the gradient
/// of a sum is the sum of gradients, the parallel result equals the serial minibatch result.
///
/// The fan-out runs through EvalApp's resource-gated, adaptively-tuned <c>ForEach</c> (CPU concurrency tuned to the
/// machine) when a license key is available; otherwise it falls back to the in-box TPL — either way multithreaded, the
/// answers identical. A default license key is SHIPPED (<see cref="DefaultEvalAppKey"/>) — EvalApp is the author's own
/// product, embedded here by permission — so the tuned fan-out is on out of the box; override per-instance or via the
/// <c>PRISM_EVALAPP_KEY</c> environment variable.
/// </summary>
public sealed class PrismTrainer : IBatchTrainer
{
    /// <summary>The shipped EvalApp license (EvalApp is the author's own product; embedded by permission).</summary>
    public const string DefaultEvalAppKey = "20270312-gwZ8hyAovecW9DmRm_OQ13xOG7oCZWvyBkYHzy_ZS8k";

    private readonly AlgFormer _model;
    private readonly int _parallelism;
    private readonly ICompiledPipeline<BatchData>? _pipeline;   // EvalApp fan-out; null → in-box TPL

    public PrismTrainer(AlgFormer model, int parallelism = 0, string? evalAppKey = null)
    {
        _model = model;
        _parallelism = parallelism > 0 ? parallelism : Math.Max(1, Environment.ProcessorCount - 1);
        var key = evalAppKey ?? Environment.GetEnvironmentVariable("PRISM_EVALAPP_KEY") ?? DefaultEvalAppKey;
        if (!string.IsNullOrWhiteSpace(key))
        {
            try { _pipeline = BuildPipeline(key); }
            catch { _pipeline = null; }   // any API/license mismatch → in-box TPL, never fail training
        }
    }

    /// <summary>Whether the tuned EvalApp fan-out is active (else in-box TPL).</summary>
    public bool UsingEvalApp => _pipeline is not null;

    /// <summary>One data-parallel minibatch step. Returns mean CE loss. Result is identical to a serial batch.</summary>
    public double TrainBatch(IReadOnlyList<(int[] Ctx, int Target)> batch, double lr = 1e-3, CancellationToken ct = default)
    {
        if (batch.Count == 0) return 0;
        try
        {
            var (merged, loss) = _pipeline is not null ? RunEvalApp(batch, ct) : RunInBox(batch, ct);
            _model.Step(merged, lr, scale: batch.Count);   // mean-gradient
            return loss / batch.Count;
        }
        catch (OperationCanceledException) { return 0; }   // Stop pressed mid-batch — discard the partial, land promptly
    }

    /// <summary>Accumulate a batch's gradient in parallel WITHOUT applying it — for a distributed coordinator to merge
    /// this (local) shard with gradients arriving from networked slaves before a single Step.</summary>
    public AlgFormer.Grads AccumulateBatch(IReadOnlyList<(int[] Ctx, int Target)> batch, out double meanLoss)
    {
        if (batch.Count == 0) { meanLoss = 0; return _model.NewGrads(); }
        var (merged, loss) = _pipeline is not null ? RunEvalApp(batch, default) : RunInBox(batch, default);
        meanLoss = loss / batch.Count; return merged;
    }

    /// <summary>Train one pass over the data in minibatches (shuffled). Returns mean loss.</summary>
    public double TrainEpoch(IReadOnlyList<(int[] Ctx, int Target)> data, int batchSize, double lr, int shuffleSeed)
    {
        var order = Enumerable.Range(0, data.Count).ToArray();
        var rng = new Random(shuffleSeed);
        for (var i = order.Length - 1; i > 0; i--) { var j = rng.Next(i + 1); (order[i], order[j]) = (order[j], order[i]); }
        double total = 0; var batches = 0;
        for (var start = 0; start < order.Length; start += batchSize)
        {
            var end = Math.Min(start + batchSize, order.Length);
            var batch = new (int[], int)[end - start];
            for (var k = start; k < end; k++) batch[k - start] = data[order[k]];
            total += TrainBatch(batch); batches++;
        }
        return batches > 0 ? total / batches : 0;
    }

    /// <summary>Train N epochs with linear LR decay (base → 0.1×) — the schedule that keeps the multiplicative FFN
    /// stable at scale (a fixed high LR diverges).</summary>
    public void Train(IReadOnlyList<(int[] Ctx, int Target)> data, int epochs, int batchSize = 256, double baseLr = 1e-3, int seed = 1, Action<int, double>? onEpoch = null)
    {
        for (var ep = 1; ep <= epochs; ep++)
        {
            var lr = baseLr * (1.0 - 0.9 * (ep - 1) / Math.Max(1, epochs - 1));
            var loss = TrainEpoch(data, batchSize, lr, seed + ep);
            onEpoch?.Invoke(ep, loss);
        }
    }

    // ---- shard split (shared by both backends) ----
    private List<(int[] Ctx, int Target)>[] Split(IReadOnlyList<(int[] Ctx, int Target)> batch)
    {
        var p = Math.Min(_parallelism, batch.Count);
        var shards = new List<(int[], int)>[p];
        for (var i = 0; i < p; i++) shards[i] = new List<(int[], int)>();
        for (var i = 0; i < batch.Count; i++) shards[i % p].Add(batch[i]);
        return shards;
    }

    private (AlgFormer.Grads, double) AccumulateShard(List<(int[] Ctx, int Target)> shard, CancellationToken ct)
    {
        var g = _model.NewGrads(); var loss = 0.0; var scratch = _model.NewScratch();   // one row-pool per shard (thread), reused across this shard's examples → ~0 alloc after warmup
        foreach (var (ctx, tgt) in shard) { ct.ThrowIfCancellationRequested(); loss += _model.Accumulate(ctx, tgt, g, scratch); }   // per-example cancel → Stop lands after the current example
        return (g, loss);
    }

    // ---- in-box TPL backend (always available) ----
    private (AlgFormer.Grads, double) RunInBox(IReadOnlyList<(int[] Ctx, int Target)> batch, CancellationToken ct)
    {
        var shards = Split(batch);
        var results = new (AlgFormer.Grads g, double loss)[shards.Length];
        System.Threading.Tasks.Parallel.For(0, shards.Length, new ParallelOptions { CancellationToken = ct }, i => results[i] = AccumulateShard(shards[i], ct));
        var merged = results[0].g; var loss = results[0].loss;
        for (var i = 1; i < results.Length; i++) { merged.Add(results[i].g); loss += results[i].loss; }
        return (merged, loss);
    }

    // ---- EvalApp gated ForEach backend (tuned, licensed) ----
    private (AlgFormer.Grads, double) RunEvalApp(IReadOnlyList<(int[] Ctx, int Target)> batch, CancellationToken ct)
    {
        var result = _pipeline!.RunAsync(new BatchData(this, batch), ct).GetAwaiter().GetResult();
        var data = result switch
        {
            PipelineResult<BatchData>.Success s => s.Data,
            PipelineResult<BatchData>.Failure f => throw new InvalidOperationException(f.Message ?? "fan-out failed", f.Exception),
            _ => throw new InvalidOperationException("unsupported pipeline result"),
        };
        return (data.Merged!, data.Loss);
    }

    internal sealed record BatchData(PrismTrainer Trainer, IReadOnlyList<(int[] Ctx, int Target)> Batch)
    {
        public AlgFormer.Grads? Merged { get; init; }
        public double Loss { get; init; }
    }

    internal sealed class ShardItem
    {
        public required PrismTrainer Trainer;
        public required List<(int[] Ctx, int Target)> Work;
        public AlgFormer.Grads? Grads;
        public double Loss;
    }

    private sealed class AccumulateStep : IStep<ShardItem>
    {
        public ValueTask<ShardItem> ExecuteAsync(ShardItem item, CancellationToken ct)
        {
            var (g, loss) = item.Trainer.AccumulateShard(item.Work, ct);   // per-example cancel inside
            item.Grads = g; item.Loss = loss;
            return ValueTask.FromResult(item);
        }
    }

    private static IEnumerable<ShardItem> SelectShards(BatchData d) =>
        d.Trainer.Split(d.Batch).Select(s => new ShardItem { Trainer = d.Trainer, Work = s });

    private static BatchData MergeShards(BatchData d, IReadOnlyList<ShardItem> items)
    {
        var merged = d.Trainer._model.NewGrads(); var loss = 0.0;
        foreach (var it in items) { if (it.Grads is not null) merged.Add(it.Grads); loss += it.Loss; }
        return d with { Merged = merged, Loss = loss };
    }

    private ICompiledPipeline<BatchData> BuildPipeline(string key)
    {
        Eval.App("PrismFormer")
            .WithContext(NullGlobalContext.Instance)
            .WithResource(ResourceKind.Cpu)
            .WithTuning()
            .DefineDomain("PrismFormer", new object())
                .DefineTask<BatchData>("TrainBatch")
                    .Gate(ResourceKind.Cpu, null, g => g
                        .ForEach(SelectShards, MergeShards, "Shards", Tunable.ForItems(), sub => sub.AddStep("Accumulate", new AccumulateStep())))
                    .Run(out var pipeline)
                .Build(key);
        return pipeline;
    }
}
