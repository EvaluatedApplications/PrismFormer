// Copyright (c) 2026 Dongyang Stephen Chen, trading as Evaluated Applications. All rights reserved.
// Source-available for study and verification; commercial use, redistribution, or derivatives require permission. See LICENSE.

using System.Diagnostics;
using PrismFormer;

namespace PrismFormer.Bench;

/// <summary>
/// Perf sweep of the two IDENTITY-PRESERVING upgrade-in-place levers — grow Shifts (S, relation-bank rank) and grow
/// Context (the window). Measures, at production depth (L8, D256), the COMPUTE cost of each config: parameter count +
/// memory, per-token serve latency (KV cache) at a fixed workload, per-token serve at a NEAR-FULL-context workload
/// (exposes the O(T²) attention cost of a longer window), and per-example train-step time. Prints a cost table + deltas
/// vs the base so the cost/benefit of each upgrade is explicit. Usage: prismformer-bench --upgrade.
/// </summary>
public static class UpgradeBench
{
    private readonly record struct Cfg(string Name, int S, int C);

    public static void Run()
    {
        try { Console.OutputEncoding = System.Text.Encoding.UTF8; } catch { }
        const int vocab = 96, L = 8;
        var D = PhasorCodec.Dim; var frozen = PhasorCodec.FrozenReals;
        var rng = new Random(1);

        var cfgs = new[]
        {
            new Cfg("base S64 C256", 64, 256),   // the current PRISM-2 spec
            new Cfg("S96  C256", 96, 256),
            new Cfg("S128 C256", 128, 256),
            new Cfg("S64  C512", 64, 512),
            new Cfg("S128 C512", 128, 512),
        };

        int[] Rand(int n) => Enumerable.Range(0, n).Select(_ => rng.Next(vocab)).ToArray();

        // per-token serve latency (KV cache): prime `prime` tokens, then time `gen` incremental Steps.
        double ServeMsPerTok(AlgFormer m, int prime, int gen, int cap)
        {
            var toks = Rand(prime);
            { var w = m.NewCache(); m.Prime(w, toks); for (var k = 0; k < 4 && w.Length < cap; k++) m.Step(w, rng.Next(vocab)); }   // warm JIT
            var c = m.NewCache(); m.Prime(c, toks);
            var sw = Stopwatch.StartNew();
            var n = 0; for (var k = 0; k < gen && c.Length < cap; k++) { m.Step(c, rng.Next(vocab)); n++; }
            sw.Stop();
            return n > 0 ? sw.Elapsed.TotalMilliseconds / n : 0;
        }

        Console.WriteLine($"PrismFormer UPGRADE-IN-PLACE perf sweep   {DateTime.Now:yyyy-MM-dd HH:mm}   (L{L}, D{D}, vocab {vocab})");
        Console.WriteLine("serve = prime 128 + generate 64 (fixed workload, isolates the S cost)");
        Console.WriteLine("long  = prime near-full-context + generate 32 (exposes the O(T²) cost of a longer window)");
        Console.WriteLine("train = one Accumulate + Adam Step on a 200-token example (avg of 4), single-thread\n");
        Console.WriteLine($"  {"config",-15} {"params",11} {"MB",6} {"serve ms/tok",13} {"long ms/tok",12} {"train ms/step",14}");
        Console.WriteLine("  " + new string('-', 74));

        long baseP = 0; double baseServe = 0, baseLong = 0, baseTrain = 0;
        foreach (var c in cfgs)
        {
            var m = new AlgFormer(vocab, shifts: c.S, layers: L, maxContext: c.C, dModel: D, frozenPrefix: frozen, seed: 1);
            var p = m.ParamCount; var mb = p * 8.0 / 1e6;

            var serve = ServeMsPerTok(m, prime: 128, gen: 64, cap: c.C);
            var lng = ServeMsPerTok(m, prime: c.C - 16, gen: 32, cap: c.C);

            var tctx = Math.Min(200, c.C - 1);
            { var g0 = m.NewGrads(); m.Accumulate(Rand(tctx), rng.Next(vocab), g0); m.Step(g0, 1e-3); }   // warm
            var tw = 0.0;
            for (var r = 0; r < 4; r++) { var g = m.NewGrads(); var s = Stopwatch.StartNew(); m.Accumulate(Rand(tctx), rng.Next(vocab), g); m.Step(g, 1e-3); s.Stop(); tw += s.Elapsed.TotalMilliseconds; }
            var train = tw / 4;

            if (c.Name.StartsWith("base")) { baseP = p; baseServe = serve; baseLong = lng; baseTrain = train; }
            Console.WriteLine($"  {c.Name,-15} {p,11:N0} {mb,6:F1} {serve,13:F2} {lng,12:F2} {train,14:F1}");
        }

        Console.WriteLine("\ncost vs base (×) ---------------------------------------------------------");
        Console.WriteLine($"  {"config",-15} {"params",8} {"serve",8} {"long",8} {"train",8}");
        foreach (var c in cfgs)
        {
            var m = new AlgFormer(vocab, shifts: c.S, layers: L, maxContext: c.C, dModel: D, frozenPrefix: frozen, seed: 1);
            var p = m.ParamCount;
            var serve = ServeMsPerTok(m, 128, 64, c.C);
            var lng = ServeMsPerTok(m, c.C - 16, 32, c.C);
            var tctx = Math.Min(200, c.C - 1);
            var tw = 0.0; for (var r = 0; r < 4; r++) { var g = m.NewGrads(); var s = Stopwatch.StartNew(); m.Accumulate(Rand(tctx), rng.Next(vocab), g); m.Step(g, 1e-3); s.Stop(); tw += s.Elapsed.TotalMilliseconds; }
            var train = tw / 4;
            Console.WriteLine($"  {c.Name,-15} {(double)p / baseP,7:F2}x {serve / baseServe,7:F2}x {lng / baseLong,7:F2}x {train / baseTrain,7:F2}x");
        }
        Console.WriteLine("\nnotes: Shifts (S) scales bank params + serve + train ~linearly (S is the lean↔full knob; S=D recovers a full matrix).");
        Console.WriteLine("       Context adds only Pos rows (few params) — its cost is the longer window's O(T²) attention, seen in 'long', not 'serve'.");
    }

    // ── The BENEFIT half: train each config on char-level next-token prediction for a FIXED wall-clock budget, then
    //    measure held-out next-char accuracy + bits/char. The larger-Context configs get to USE the longer window. ──
    private static string LoadCorpus(int maxChars)
    {
        var dir = AppContext.BaseDirectory;
        for (var i = 0; i < 9 && dir != null; i++)
        {
            var t = Path.Combine(dir, "studio", "PrismStudio", "data", "text");
            if (Directory.Exists(t))
            {
                var sb = new System.Text.StringBuilder();
                foreach (var f in Directory.EnumerateFiles(t, "*.txt").OrderBy(x => x)) { sb.Append(File.ReadAllText(f)); if (sb.Length >= maxChars) break; }
                if (sb.Length > 100) return sb.ToString(0, Math.Min(sb.Length, maxChars));
            }
            dir = Directory.GetParent(dir)?.FullName;
        }
        return string.Concat(Enumerable.Repeat("the quick brown fox jumps over the lazy dog. she sells sea shells by the sea shore. ", 3000));   // fallback
    }

    public static void RunLm(int secondsPerConfig = 60)
    {
        try { Console.OutputEncoding = System.Text.Encoding.UTF8; } catch { }
        const int vocab = CharVocab.N, L = 8;
        var D = PhasorCodec.Dim; var frozen = PhasorCodec.FrozenReals;
        var v = new CharVocab();
        double[] Seed(int w) => w < CharVocab.Printable ? PhasorCodec.Encode(((char)(32 + w)).ToString()) : new double[PhasorCodec.Dim];

        var raw = LoadCorpus(700_000);
        var norm = new string(raw.Select(c => c >= 32 && c <= 126 ? c : ' ').ToArray());
        var ids = v.Encode(norm);
        var split = (int)(ids.Length * 0.9);
        var train = ids[..split]; var held = ids[split..];

        var cfgs = new[] { new Cfg("base S64 C256", 64, 256), new Cfg("S128 C256", 128, 256), new Cfg("S64 C512", 64, 512), new Cfg("S128 C512", 128, 512) };
        var rng = new Random(1);

        (double acc, double bpc) Eval(AlgFormer m, int C, int samples)
        {
            int ok = 0, n = 0; double bits = 0; var stride = Math.Max(1, (held.Length - C - 1) / samples);
            for (var p = C; p < held.Length - 1; p += stride)
            {
                var ctx = held[(p - C)..p];
                var lg = m.LogitsFor(ctx);
                var mx = double.NegativeInfinity; for (var i = 0; i < lg.Length; i++) if (lg[i] > mx) mx = lg[i];
                var best = 0; for (var i = 1; i < lg.Length; i++) if (lg[i] > lg[best]) best = i;
                if (best == held[p]) ok++;
                var sum = 0.0; for (var i = 0; i < lg.Length; i++) sum += Math.Exp(lg[i] - mx);
                bits += -(lg[held[p]] - mx - Math.Log(sum)) / Math.Log(2); n++;
            }
            return (n > 0 ? ok / (double)n : 0, n > 0 ? bits / n : 0);
        }

        Console.WriteLine($"PrismFormer UPGRADE benefit — char LM next-token   {DateTime.Now:yyyy-MM-dd HH:mm}   (L{L}, D{D})");
        Console.WriteLine($"corpus {ids.Length:N0} chars (train {train.Length:N0} / held {held.Length:N0}) · {secondsPerConfig}s train budget per config\n");
        Console.WriteLine($"  {"config",-15} {"params",11} {"epochs",7} {"held next-char",15} {"bits/char",10}");
        Console.WriteLine("  " + new string('-', 62));

        foreach (var c in cfgs)
        {
            var m = new AlgFormer(vocab, shifts: c.S, layers: L, maxContext: c.C, dModel: D, frozenPrefix: frozen, embedSeed: Seed, seed: 1);
            // fixed set of full-context windows from the train slice
            var data = new List<(int[] Ctx, int Target)>(2000);
            for (var i = 0; i < 2000; i++) { var p = c.C + rng.Next(train.Length - c.C - 1); data.Add((train[(p - c.C)..p], train[p])); }
            var sw = Stopwatch.StartNew(); var ep = 0;
            while (sw.Elapsed.TotalSeconds < secondsPerConfig) { m.Train(data, 1, batchSize: 64, baseLr: 1.5e-3, seed: 1 + ep); ep++; }
            var (acc, bpc) = Eval(m, c.C, 800);
            Console.WriteLine($"  {c.Name,-15} {m.ParamCount,11:N0} {ep,7} {acc,14:P1} {bpc,10:F3}");
        }
        Console.WriteLine("\nread: higher next-char % / lower bits/char = better. Compare C256 vs C512 (does the longer window help next-token?)");
        Console.WriteLine("      and S64 vs S128 (does more capacity help?). 'epochs' = how many passes each fit in the fixed budget (the cost).");
    }

    // ── Coherence probe for a checkpoint file: load it (upgrading if older) and greedily generate a few completions, so a
    //    TRAINED model (coherent text) is visibly distinguishable from a FRESH one (repeated-letter garbage). ──
    public static void RunSample(string[] paths)
    {
        var v = new CharVocab();
        double[] Seed(int w) => w < CharVocab.Printable ? PhasorCodec.Encode(((char)(32 + w)).ToString()) : new double[PhasorCodec.Dim];
        foreach (var path in paths)
        {
            if (!File.Exists(path)) { Console.WriteLine($"{path}  (missing)"); continue; }
            var m = new AlgFormer(CharVocab.N, PrismSpec.Shifts, PrismSpec.Layers, PrismSpec.Context, PhasorCodec.Dim, PhasorCodec.FrozenReals, Seed, PrismSpec.InitSeed);
            var ok = false; var sig = "?";
            try
            {
                using var r = new BinaryReader(File.OpenRead(path));
                sig = r.ReadString();
                if (sig == PrismSpec.Signature) ok = m.Load(r);
                else { var old = PrismSpec.Parse(sig); ok = old != null && PrismSpec.CanUpgradeFrom(old) && m.LoadUpgrade(r, old.Context); }
            }
            catch (Exception e) { Console.WriteLine($"  load error: {e.Message}"); }
            Console.WriteLine($"\n{System.IO.Path.GetFileName(path)}   sig={sig}   loaded={ok}");
            foreach (var prompt in new[] { "the ", "hello ", "1 + 1 = ", "user: hi\nprism: " })
            {
                var gen = m.Generate(v.Encode(prompt), 24);
                Console.WriteLine($"    \"{prompt.Replace("\n", "\\n")}\" -> \"{v.Decode(gen).Replace("\n", "\\n")}\"");
            }
        }
    }

    // ── Bit-identity guard for the backprop hot path: run a fixed, seeded sequence of Accumulate calls and print an EXACT
    //    (bitwise FNV) checksum of every gradient. Any change to ForwardAll/AlgBack/Accumulate that alters a single bit
    //    changes this hash — so the scratch-reuse refactor must leave it UNCHANGED. Also prints loss (readable sanity). ──
    public static void RunGradcheck()
    {
        var v = new CharVocab();
        double[] Seed(int w) => w < CharVocab.Printable ? PhasorCodec.Encode(((char)(32 + w)).ToString()) : new double[PhasorCodec.Dim];
        var m = new AlgFormer(CharVocab.N, shifts: PrismSpec.Shifts, layers: PrismSpec.Layers, maxContext: 128, dModel: PhasorCodec.Dim, frozenPrefix: PhasorCodec.FrozenReals, embedSeed: Seed, seed: 1);
        var rng = new Random(7);
        var g = m.NewGrads();
        var loss = 0.0;
        var scratch = m.NewScratch();   // ONE pool reused across all examples — exercises the pooled backprop path
        // varied lengths on purpose — a reused scratch must be re-cleared between a long example and a shorter one
        foreach (var len in new[] { 40, 96, 12, 64, 96, 8, 120, 33 })
        {
            var ctx = Enumerable.Range(0, len).Select(_ => rng.Next(CharVocab.N)).ToArray();
            loss += m.Accumulate(ctx, rng.Next(CharVocab.N), g, scratch);
        }
        Console.WriteLine($"gradcheck: loss={loss:R}  checksum={m.GradSignature(g):X16}");
    }

    // ── Training throughput/utilization profiler: run the REAL production training path (PrismTrainer, EvalApp-gated
    //    fan-out) for a fixed budget and report effective cores busy, batches/s, and GC pressure — to see why the CPU
    //    doesn't saturate. Realistic short examples (most pairs/chat are short) against the production model shape. ──
    public static void RunProfile(int seconds = 20, int ctxLen = 180, int batchSize = 64)
    {
        try { Console.OutputEncoding = System.Text.Encoding.UTF8; } catch { }
        var v = new CharVocab();
        double[] Seed(int w) => w < CharVocab.Printable ? PhasorCodec.Encode(((char)(32 + w)).ToString()) : new double[PhasorCodec.Dim];
        var model = new AlgFormer(CharVocab.N, shifts: PrismSpec.Shifts, layers: PrismSpec.Layers, maxContext: PrismSpec.Context, dModel: PhasorCodec.Dim, frozenPrefix: PhasorCodec.FrozenReals, embedSeed: Seed, seed: 1);
        var trainer = new PrismTrainer(model);
        var rng = new Random(1);
        var data = Enumerable.Range(0, 4096).Select(_ => (Enumerable.Range(0, ctxLen).Select(__ => rng.Next(CharVocab.N)).ToArray(), rng.Next(CharVocab.N))).ToList();

        Console.WriteLine($"PrismFormer TRAINING PROFILE   {DateTime.Now:yyyy-MM-dd HH:mm}");
        Console.WriteLine($"cores={Environment.ProcessorCount}  UsingEvalApp={trainer.UsingEvalApp}  ServerGC={System.Runtime.GCSettings.IsServerGC}  ctx={ctxLen}  batch={batchSize}  (model L{PrismSpec.Layers}/D{PhasorCodec.Dim}/S{PrismSpec.Shifts}/c{PrismSpec.Context})");

        trainer.TrainBatch(data.Take(batchSize).ToList());   // warm JIT

        var proc = System.Diagnostics.Process.GetCurrentProcess();
        var alloc0 = GC.GetTotalAllocatedBytes(true);
        int g00 = GC.CollectionCount(0), g10 = GC.CollectionCount(1), g20 = GC.CollectionCount(2);
        var cpu0 = proc.TotalProcessorTime;
        var sw = Stopwatch.StartNew(); var batches = 0; long examples = 0; var idx = 0;
        while (sw.Elapsed.TotalSeconds < seconds)
        {
            var batch = new (int[], int)[batchSize];
            for (var k = 0; k < batchSize; k++) batch[k] = data[idx++ % data.Count];
            trainer.TrainBatch(batch); batches++; examples += batchSize;
        }
        sw.Stop();
        var cpu1 = proc.TotalProcessorTime;
        var alloc1 = GC.GetTotalAllocatedBytes(true);
        var wall = sw.Elapsed.TotalSeconds;
        var eff = (cpu1 - cpu0).TotalSeconds / wall;

        Console.WriteLine($"\n  batches/s          = {batches / wall,8:F1}   examples/s = {examples / wall:F0}");
        Console.WriteLine($"  effective cores    = {eff,8:F2}   of {Environment.ProcessorCount}  ({100 * eff / Environment.ProcessorCount:F0}% CPU)");
        Console.WriteLine($"  GC gen0/1/2        = {GC.CollectionCount(0) - g00}/{GC.CollectionCount(1) - g10}/{GC.CollectionCount(2) - g20}");
        Console.WriteLine($"  allocated          = {(alloc1 - alloc0) / 1e9,8:F2} GB   ({(alloc1 - alloc0) / 1e9 / wall:F2} GB/s)");
        Console.WriteLine(eff < Environment.ProcessorCount * 0.7 ? "\n  -> under-utilized: CPU gate default (cores/2) and/or Workstation GC stalls are throttling the fan-out." : "\n  -> saturating.");
    }

    // ── Verify the upgrade-in-place round-trip BEFORE shipping a spec bump to the colony: a checkpoint saved at the old
    //    Context must LoadUpgrade into the new Context byte-clean — identical logits on any in-window prompt (weights carry
    //    over, new Pos rows zero-pad → contribute nothing). If this drifts, every node's checkpoint would corrupt on load. ──
    public static void RunSpec(int oldContext = 512)
    {
        try { Console.OutputEncoding = System.Text.Encoding.UTF8; } catch { }
        const int vocab = CharVocab.N, L = 8, S = 64;
        var D = PhasorCodec.Dim; var frozen = PhasorCodec.FrozenReals;
        var v = new CharVocab();
        double[] Seed(int w) => w < CharVocab.Printable ? PhasorCodec.Encode(((char)(32 + w)).ToString()) : new double[PhasorCodec.Dim];

        Console.WriteLine($"current spec Signature = {PrismSpec.Signature}");
        var oldSig = $"{PrismSpec.Version}/v{vocab}/d{D}/f{frozen}/c{oldContext}/L{L}/S{S}";
        var old = PrismSpec.Parse(oldSig);
        Console.WriteLine($"old-on-disk Signature  = {oldSig}");
        Console.WriteLine($"CanUpgradeFrom(old)    = {(old != null && PrismSpec.CanUpgradeFrom(old))}");

        // "old" model at the old context, given some non-trivial training so weights aren't all-init
        var a = new AlgFormer(vocab, shifts: S, layers: L, maxContext: oldContext, dModel: D, frozenPrefix: frozen, embedSeed: Seed, seed: 1);
        var corpus = v.Encode(new string(LoadCorpus(60_000).Select(c => c >= 32 && c <= 126 ? c : ' ').ToArray()));
        var rng = new Random(1);
        var data = Enumerable.Range(0, 300).Select(_ => { var p = 40 + rng.Next(corpus.Length - 41); return (corpus[(p - 40)..p], corpus[p]); }).ToList();
        a.Train(data, 2, batchSize: 64, baseLr: 1.5e-3, seed: 1);

        var prompt = v.Encode("user: hello there\nprism: ");   // short, within the old window
        var la = a.LogitsFor(prompt);

        var ms = new MemoryStream();
        using (var w = new BinaryWriter(ms, System.Text.Encoding.UTF8, true)) a.Save(w);
        ms.Position = 0;
        var b = new AlgFormer(vocab, shifts: S, layers: L, maxContext: PrismSpec.Context, dModel: D, frozenPrefix: frozen, embedSeed: Seed, seed: 1);
        bool okUp; using (var r = new BinaryReader(ms, System.Text.Encoding.UTF8, true)) okUp = b.LoadUpgrade(r, oldContext);
        var lb = b.LogitsFor(prompt);

        var maxDiff = 0.0; for (var i = 0; i < la.Length; i++) maxDiff = Math.Max(maxDiff, Math.Abs(la[i] - lb[i]));
        Console.WriteLine($"LoadUpgrade ok         = {okUp}   ({old!.Context} -> {PrismSpec.Context})");
        Console.WriteLine($"identity check         = max|logit_old - logit_new| = {maxDiff:E3}  (in-window prompt -> must be ~0)");
        Console.WriteLine(okUp && maxDiff < 1e-9 ? "  PASS — upgrade is identity-preserving; c256 checkpoints carry over byte-clean, no retrain." : "  FAIL — drift or load error; DO NOT ship the bump.");
    }
}
