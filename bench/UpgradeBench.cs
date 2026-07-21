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
        double[] Seed(int w) => PhasorCodec.Encode(v.Symbol(w));   // char OR subword face
        foreach (var path in paths)
        {
            if (!File.Exists(path)) { Console.WriteLine($"{path}  (missing)"); continue; }
            var m = new AlgFormer(PrismSpec.Vocab, PrismSpec.Shifts, PrismSpec.Layers, PrismSpec.Context, PhasorCodec.Dim, PhasorCodec.FrozenReals, Seed, PrismSpec.InitSeed);
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
    public static void RunSpec(int oldContext = 512, int oldShifts = 64)
    {
        try { Console.OutputEncoding = System.Text.Encoding.UTF8; } catch { }
        const int oldVocab = CharVocab.N, L = 8;   // "old on disk" = the char-level model before the bump
        var D = PhasorCodec.Dim; var frozen = PhasorCodec.FrozenReals;
        var v = new CharVocab();
        double[] Seed(int w) => PhasorCodec.Encode(v.Symbol(w));   // char OR subword face — valid for both the v96 old and the v-full new model
        int[] Chars(string s) => s.Select(c => v.Id(c)).ToArray();  // CHAR ids only, so the v96 old model never sees a subword id

        Console.WriteLine($"current spec Signature = {PrismSpec.Signature}");
        var oldSig = $"{PrismSpec.Version}/v{oldVocab}/d{D}/f{frozen}/c{oldContext}/L{L}/S{oldShifts}";
        var old = PrismSpec.Parse(oldSig);
        Console.WriteLine($"old-on-disk Signature  = {oldSig}");
        Console.WriteLine($"CanUpgradeFrom(old)    = {(old != null && PrismSpec.CanUpgradeFrom(old))}");

        // "old" char-level model at the old context/shifts, given some non-trivial training so weights aren't all-init
        var a = new AlgFormer(oldVocab, shifts: oldShifts, layers: L, maxContext: oldContext, dModel: D, frozenPrefix: frozen, embedSeed: Seed, seed: 1);
        var corpus = Chars(new string(LoadCorpus(60_000).Select(c => c >= 32 && c <= 126 ? c : ' ').ToArray()));
        var rng = new Random(1);
        var data = Enumerable.Range(0, 300).Select(_ => { var p = 40 + rng.Next(corpus.Length - 41); return (corpus[(p - 40)..p], corpus[p]); }).ToList();
        a.Train(data, 2, batchSize: 64, baseLr: 1.5e-3, seed: 1);

        var prompt = Chars("user: hello there\nprism: ");   // CHAR ids — valid for both old (v96) and new (v-full)
        var la = a.LogitsFor(prompt);

        var ms = new MemoryStream();
        using (var w = new BinaryWriter(ms, System.Text.Encoding.UTF8, true)) a.Save(w);
        ms.Position = 0;
        var b = new AlgFormer(PrismSpec.Vocab, shifts: PrismSpec.Shifts, layers: PrismSpec.Layers, maxContext: PrismSpec.Context, dModel: D, frozenPrefix: frozen, embedSeed: Seed, seed: PrismSpec.InitSeed);
        bool okUp; using (var r = new BinaryReader(ms, System.Text.Encoding.UTF8, true)) okUp = b.LoadUpgrade(r, oldContext);
        var lb = b.LogitsFor(prompt);

        // compare over the OLD vocab range (the preserved region): char rows + forward must be byte-identical, and the
        // new subword/shift rows must not perturb a char-token forward (subword rows unused, new shift rows are zero).
        var maxDiff = 0.0; for (var i = 0; i < oldVocab; i++) maxDiff = Math.Max(maxDiff, Math.Abs(la[i] - lb[i]));
        Console.WriteLine($"LoadUpgrade ok         = {okUp}   (v{oldVocab}/c{oldContext}/S{oldShifts} -> v{PrismSpec.Vocab}/c{PrismSpec.Context}/S{PrismSpec.Shifts})");
        Console.WriteLine($"identity check         = max|logit_old - logit_new| over char range = {maxDiff:E3}  (char knowledge must carry over -> ~0)");
        Console.WriteLine(okUp && maxDiff < 1e-9 ? "  PASS — char knowledge carries over byte-clean; new shift/subword rows are identity at init. Safe to ship." : "  FAIL — drift or load error; DO NOT ship the bump.");
    }

    // ── EXPERIMENT: is a LAYER add non-destructive AND trainable? A naive all-zero layer is identity at init but its
    //    gradient is exactly zero everywhere (ctx=z=0) → DEAD, never trains. Zeroing ONLY the residual output projections
    //    (Ro, Ao) is also identity at init but ctx/z are nonzero → LIVE gradient, so it trains and can help. This measures
    //    all of that head-to-head against a same-budget base control (char LM). ──
    public static void RunGrowLayer()
    {
        try { Console.OutputEncoding = System.Text.Encoding.UTF8; } catch { }
        const int vocab = CharVocab.N, S = 48, baseL = 2, ctx = 48; var D = PhasorCodec.Dim; var frozen = PhasorCodec.FrozenReals;
        var v = new CharVocab();
        double[] Seed(int w) => PhasorCodec.Encode(v.Symbol(w));
        int[] Chars(string s) => s.Select(c => c >= 32 && c <= 126 ? c - 32 : 0).ToArray();   // char ids 0..94 (CharVocab.Id), no subwords
        var corpus = Chars(LoadCorpus(120_000));
        List<(int[], int)> Draw(int n, int seed) { var r = new Random(seed); return Enumerable.Range(0, n).Select(_ => { var p = ctx + r.Next(corpus.Length - ctx - 1); return (corpus[(p - ctx)..p], corpus[p]); }).ToList(); }
        var evalSet = Draw(1000, 99);
        double Loss(AlgFormer m) { double s = 0; foreach (var (c, t) in evalSet) { var lg = m.LogitsFor(c); var mx = lg.Max(); double sum = 0; foreach (var x in lg) sum += Math.Exp(x - mx); s += -(lg[t] - mx - Math.Log(sum)); } return s / evalSet.Count; }
        double MaxDiff(double[] a, double[] b) { double d = 0; for (var i = 0; i < a.Length; i++) d = Math.Max(d, Math.Abs(a[i] - b[i])); return d; }

        Console.WriteLine($"NON-DESTRUCTIVE LAYER ADD — char LM, base L{baseL} -> L{baseL + 1}   (d{D} S{S} ctx{ctx})   {DateTime.Now:yyyy-MM-dd HH:mm}");
        var baseM = new AlgFormer(vocab, S, baseL, ctx, D, frozen, Seed, 1);
        baseM.Train(Draw(4000, 1), 8, batchSize: 64, baseLr: 2e-3, seed: 1);
        var probe = corpus[200..(200 + ctx)]; var baseLogits = baseM.LogitsFor(probe);
        Console.WriteLine($"  base trained: eval loss {Loss(baseM):F4}");

        var B = baseM.GrowLayers(1, zeroOutputOnly: true, seed: 7);    // zero ONLY Ro/Ao → identity + LIVE
        var C = baseM.GrowLayers(1, zeroOutputOnly: false, seed: 7);   // zero the WHOLE layer → identity + DEAD (control)
        var ctrl = baseM.GrowLayers(0);                                // same params, no new layer → fair same-budget control

        Console.WriteLine("verification (at init):");
        Console.WriteLine($"    identity  B: max|logit-base| = {MaxDiff(baseLogits, B.LogitsFor(probe)):E2}   C: {MaxDiff(baseLogits, C.LogitsFor(probe)):E2}   (both must be ~0)");
        Console.WriteLine($"    new-layer output-bank norm  B: {B.OutputBankNorm(baseL):E2}   C: {C.OutputBankNorm(baseL):E2}   (both 0 at init)");

        var phase2 = Draw(4000, 2);
        B.Train(phase2, 8, batchSize: 64, baseLr: 2e-3, seed: 2);
        C.Train(phase2, 8, batchSize: 64, baseLr: 2e-3, seed: 2);
        ctrl.Train(phase2, 8, batchSize: 64, baseLr: 2e-3, seed: 2);

        var bN = B.OutputBankNorm(baseL); var cN = C.OutputBankNorm(baseL);
        double bl = Loss(B), cl = Loss(C), ctl = Loss(ctrl);
        Console.WriteLine("after equal-budget training:");
        Console.WriteLine($"    new-layer output-bank norm  B: {bN:E2} ({(bN > 1e-6 ? "LIVE — it trained" : "DEAD")})   C: {cN:E2} ({(cN > 1e-6 ? "live" : "DEAD — never moved")})");
        Console.WriteLine($"    eval loss   base+layer(live) {bl:F4}   |  base+layer(dead) {cl:F4}   |  base-only ctrl {ctl:F4}");
        Console.WriteLine("\nreads:");
        Console.WriteLine($"  - all-zero layer is a NO-OP: dead norm {cN:E2}, and its loss {cl:F4} ~= base ctrl {ctl:F4} (the layer changed nothing).");
        Console.WriteLine($"  - zero-output layer is LIVE: norm grew {B.OutputBankNorm(baseL):E2} from 0, loss {bl:F4} vs ctrl {ctl:F4} ({(bl < ctl ? "BETTER — the added depth helped" : "not better this budget")}).");
        var pass = bN > 1e-6 && cN < 1e-9;
        Console.WriteLine(pass ? "  VERDICT: non-destructive add works — identity at init, live gradient, trains; the all-zero control is provably dead." : "  VERDICT: unexpected — check the init.");
    }
}
