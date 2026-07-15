// Copyright (c) 2026 Dongyang Stephen Chen, trading as Evaluated Applications. All rights reserved.
// Source-available for study and verification; commercial use, redistribution, or derivatives require permission. See LICENSE.

using EvalApp.Consumer;

namespace PrismFormer;

/// <summary>
/// The ONE EvalApp application for Prism (see the refactor plan). A single licensed app registers the machine's
/// contended resources — <see cref="ResourceKind.Cpu"/> and <see cref="ResourceKind.Network"/> — under one adaptive
/// tuner, and compiles the reusable pipelines every parallel/gated operation in the library runs through:
///   • <b>Map</b>  — a CPU-gated fan-out that runs a chunk delegate in parallel (the inference forward pass uses this).
///   • <b>Send</b> — a Network-gated side effect (the MQTT relay + mesh route every outbound publish through this).
/// Training builds its own CPU-gated ForEach task on this same app (see <see cref="PrismTrainer"/>).
///
/// Nothing hand-rolls threads. When no license key is present EvalApp runs everything SEQUENTIALLY
/// (<c>LicenseMode.Unlicensed</c>), so the fallbacks below (direct/inline) only ever engage if the app failed to build.
/// A default key is SHIPPED — EvalApp is the author's own product, embedded by permission.
/// </summary>
public static class PrismEval
{
    /// <summary>The shipped EvalApp license (author's own product; embedded by permission). Override via <c>PRISM_EVALAPP_KEY</c>.</summary>
    public const string DefaultKey = "20270312-gwZ8hyAovecW9DmRm_OQ13xOG7oCZWvyBkYHzy_ZS8k";

    // ── the one app + its compiled pipelines (null → unlicensed/failed: callers fall back to inline/direct) ──
    private static readonly ICompiledPipeline<MapData>? _map;
    private static readonly ICompiledPipeline<SendData>? _send;

    /// <summary>The CPU-gated parallel map (EvalApp-tuned when licensed, else a sequential loop).</summary>
    public static IParallelMap Cpu { get; }

    /// <summary>True when the tuned EvalApp app built (else everything degrades to sequential/direct).</summary>
    public static bool Licensed => _map is not null;

    static PrismEval()
    {
        var key = Environment.GetEnvironmentVariable("PRISM_EVALAPP_KEY") ?? DefaultKey;
        try
        {
            Eval.App("PrismFormer")
                .WithContext(NullGlobalContext.Instance)
                .WithResource(ResourceKind.Cpu, Tunable.ForCpu())        // compute pool, tuned to the machine
                .WithResource(ResourceKind.Network, new TunableConfig(Min: 1, Max: 2, Default: 1))   // broker/mesh pool
                // NO .WithTuning() here: the Map fan-out is dispatched ~10k times per Serve (64 per forward × ~160 tokens).
                // Adaptive tuning re-analyses/records on EVERY RunAsync (the RunAsyncWithTuningAsync slow path), which at that
                // call count dominates wall time. The CPU gate still bounds concurrency at Tunable.ForCpu() default (= cores);
                // we just skip per-run tuning so RunAsync takes the lightweight RunAsyncCore path. (Coarse TRAINING batches,
                // which are few and long, are where adaptive tuning pays off — that pipeline can opt back in.)
                .DefineDomain("Prism")
                    .DefineTask<MapData>("Map")
                        .Gate(ResourceKind.Cpu, null, g => g
                            .ForEach(
                                (MapData d) => Enumerable.Range(0, d.Chunks).Select(i => new ChunkItem { Body = d.Body, Index = i }),
                                (MapData d, IReadOnlyList<ChunkItem> _) => d,   // side effects write disjoint slots → nothing to fold
                                "chunks", Tunable.ForCpu(),
                                sub => sub.AddStep("run", new RunChunkStep())))
                        .Run(out var map)
                    .DefineTask<SendData>("Send")
                        .Gate(ResourceKind.Network, null, g => g
                            .AddStep("publish", (IStep<SendData>)new SendStep()))   // gated on the shared Network pool
                        .Run(out var send)
                    .Build(key);
            _map = map; _send = send;
        }
        catch { _map = null; _send = null; }   // any license/API mismatch → sequential/direct, never fail
        Cpu = _map is not null ? new EvalMap() : SequentialMap.Instance;
    }

    /// <summary>Route an outbound network publish through the Network-gated pipeline (tuned/bounded concurrency).
    /// Blocking, to match the existing synchronous send sites; when unlicensed it publishes directly.</summary>
    public static void Send(Func<CancellationToken, Task> publish, CancellationToken ct = default)
    {
        if (_send is null) { try { publish(ct).GetAwaiter().GetResult(); } catch { } return; }
        try { _send.RunAsync(new SendData(publish), ct).GetAwaiter().GetResult(); } catch { }
    }

    // ── Map primitive ──────────────────────────────────────────────────────────────────────────
    /// <summary>Run <paramref name="chunks"/> invocations of <paramref name="body"/> (indices 0..chunks-1) — sequentially
    /// or across the CPU pool. The caller must ensure each chunk writes DISJOINT memory (so parallel == serial, bit-identical).</summary>
    private sealed class EvalMap : IParallelMap
    {
        public void Map(int chunks, int minForParallel, Action<int> body)
        {
            if (chunks <= 1 || chunks < minForParallel || _map is null) { for (var i = 0; i < chunks; i++) body(i); return; }
            _map.RunAsync(new MapData(body, chunks)).GetAwaiter().GetResult();
        }
    }

    internal sealed record MapData(Action<int> Body, int Chunks);
    internal sealed class ChunkItem { public required Action<int> Body; public required int Index; }
    private sealed class RunChunkStep : IStep<ChunkItem>
    {
        public ValueTask<ChunkItem> ExecuteAsync(ChunkItem item, CancellationToken ct) { item.Body(item.Index); return ValueTask.FromResult(item); }
    }

    // ── Send primitive ─────────────────────────────────────────────────────────────────────────
    internal sealed record SendData(Func<CancellationToken, Task> Publish);
    private sealed class SendStep : SideEffectStep<SendData>
    {
        public override ResourceKind? ResourceKind => EvalApp.Consumer.ResourceKind.Network;
        public override async ValueTask<SendData> ExecuteAsync(SendData d, CancellationToken ct) { await d.Publish(ct); return d; }
    }
}

/// <summary>A parallel fan-out over <c>chunks</c> disjoint index ranges — the ONLY parallelism primitive the pure
/// <see cref="AlgFormer"/> knows about. Default is sequential; <see cref="PrismEval.Cpu"/> is the EvalApp-gated impl.</summary>
public interface IParallelMap
{
    void Map(int chunks, int minForParallel, Action<int> body);
}

/// <summary>Sequential fallback — keeps <see cref="AlgFormer"/> dependency-free and is used on the training path
/// (already batch-parallel; must not nest a second fan-out).</summary>
public sealed class SequentialMap : IParallelMap
{
    public static readonly SequentialMap Instance = new();
    public void Map(int chunks, int minForParallel, Action<int> body) { for (var i = 0; i < chunks; i++) body(i); }
}
