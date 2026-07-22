// Copyright (c) 2026 Dongyang Stephen Chen, trading as Evaluated Applications. All rights reserved.
// Source-available for study and verification; commercial use, redistribution, or derivatives require permission. See LICENSE.

using System.Diagnostics;
using System.Text;
using PrismFormer;

namespace PrismFormer.Gpu;

/// <summary>
/// TEXT LANGUAGE-MODEL sizing sweep, trained on the GPU (via <see cref="GpuTrainer"/>), to ground the reset config for a
/// small CONVERSATION model: "talks fluently with a personality even if it knows nothing". The right metric is fluency,
/// NOT accuracy — held-out BITS-PER-CHARACTER (mean cross-entropy of natural language, normalised by chars so it is
/// comparable across tokenizers; lower = better) plus an actual GENERATED SAMPLE per config so coherence is read by eye.
/// Trains on the BabyLM corpus (simple developmental English, knowledge-light) with codec-seeded embeddings.
///
/// Sweeps DEPTH / WIDTH / SHIFTS / VOCAB. SHIFTS = the algebraic per-step compute unique to this arch (does it substitute
/// for size?). VOCAB = char(96) vs subword: crosstalk costs only log(V) and embeddings are codec-seeded, so a big vocab
/// may buy shorter sequences + longer effective reach for near-free — the "massive vocab for cheap" bet, measured.
/// Prints a per-config BPC curve so we can see whether the budget was enough or the config is still improving.
/// </summary>
public static class GpuText
{
    static string _text = null!;

    public static void Run(int batches = 400, int ctx = 96)
    {
        try { Console.OutputEncoding = Encoding.UTF8; } catch { }
        _text = LoadCorpus(6_000_000);
        var full = SubwordTable.List.Count;

        Console.WriteLine($"TEXT LM sizing on GPU — {GpuDevice.Describe}  (HasGpu={GpuDevice.HasGpu})");
        Console.WriteLine($"  corpus: BabyLM, {_text.Length:N0} chars · ctx {ctx} tokens · {batches} batches x128 · COARSE first pass (undertrained on purpose — read the ORDERING + curves, not absolute BPC)");
        Console.WriteLine("  metric = held-out BITS/CHAR (lower=more fluent; comparable across tokenizers) + next-token top-1 + a live sample\n");

        // (sweep, name, L, dm, S, nSubwords) — nSub=0 → char vocab (96); full = whole subword table
        var jobs = new List<(string sweep, string name, int L, int dm, int S, int nSub)>();
        jobs.Add(("BASE", "d128L6S64", 6, 128, 64, full));
        foreach (var L in new[] { 4, 8 }) jobs.Add(("DEPTH", $"L{L}", L, 128, 64, full));
        foreach (var dm in new[] { 96, 256 }) jobs.Add(("WIDTH", $"d{dm}", 6, dm, 64, full));
        jobs.Add(("SHIFTS", "S128", 6, 128, 128, full));
        jobs.Add(("VOCAB", "char96", 6, 128, 64, 0));
        jobs.Add(("VOCAB", "sub1200", 6, 128, 64, 1200));

        var sw = Stopwatch.StartNew();
        string? last = null;
        foreach (var j in jobs)
        {
            if (j.sweep != last) { Console.WriteLine($"── {j.sweep} ──"); last = j.sweep; }
            var vocab = new SubwordVocab(SubwordTable.List.Take(j.nSub).ToList());
            var (bpc, acc, prm, cpt, curve, sample) = TrainEval(vocab, j.L, j.dm, ctx, j.S, batches);
            Console.WriteLine($"   {j.name,-10} V{vocab.Size,-5} {prm / 1000.0,6:F0}k  BITS/CHAR {bpc,5:F2}  top1 {acc,4:P0}  {cpt:F1}ch/tok  ({sw.Elapsed.TotalSeconds,4:F0}s)  bpc: {curve}");
            Console.WriteLine($"       sample: \"{sample}\"\n");
        }
        Console.WriteLine($"done in {sw.Elapsed.TotalSeconds:F0}s. Read: SMALLEST config where BITS/CHAR stops dropping AND the sample reads fluent = the reset floor.");
        Console.WriteLine("Falling bpc curve at the end = wants more batches, not more size. VOCAB rows: does bigger V lower bits/char for cheap (seeded emb)?");
        GpuDevice.Shutdown();
    }

    // Train ONE config hard, printing bits/char + a live sample at each checkpoint — "can it learn to TALK at all?"
    // before we spend time sweeping sizes. Bigger batches (GPU is underused at 128) so it converges in wall-time.
    public static void RunOne(int batches = 2500, int ctx = 96, int L = 6, int dm = 128, int S = 64, int frozen = -1, bool growInit = false)
    {
        try { Console.OutputEncoding = Encoding.UTF8; } catch { }
        _text = LoadCorpus(6_000_000);
        var vocab = new SubwordVocab(SubwordTable.List.ToList());
        var V = vocab.Size;
        var toks = vocab.Encode(_text);
        var cut = (int)(toks.Length * 0.95);
        var train = toks[..cut]; var held = toks[cut..];

        double[] Seed(int w)
        {
            var f = w < V ? PhasorCodec.Encode(vocab.Symbol(w)) : new double[PhasorCodec.Dim];
            if (dm >= f.Length) return f;
            var s = new double[dm]; Array.Copy(f, s, dm); return s;
        }
        var fp = frozen >= 0 ? Math.Min(frozen, dm) : Math.Min(PhasorCodec.FrozenReals, dm);
        var codecOnly = fp >= dm;
        // growInit: build 1 layer then GrowLayers to L → layers 1..L-1 start as IDENTITY (zeroed residual output projections,
        // ReZero/Fixup-style) so a DEEP stack is trainable from step one instead of a cold all-random init.
        var cpu = growInit && L > 1
            ? new AlgFormer(V, shifts: S, layers: 1, maxContext: ctx, dModel: dm, frozenPrefix: fp, embedSeed: Seed, seed: 1).GrowLayers(L - 1, zeroOutputOnly: true)
            : new AlgFormer(V, shifts: S, layers: L, maxContext: ctx, dModel: dm, frozenPrefix: fp, embedSeed: Seed, seed: 1);
        // auto-scale GPU sub-batch so DEEP models fit (activation mem ∝ tokenBudget·L·dm); gradient identical, only more passes.
        var budget = Math.Clamp(48_000_000 / (L * dm), ctx, 24576);
        using var gt = new GpuTrainer(cpu, tokenBudget: budget);

        Console.WriteLine($"TEXT LM — CAN IT TALK? — {GpuDevice.Describe}  (HasGpu={GpuDevice.HasGpu})");
        Console.WriteLine($"  d{dm} L{L} S{S} ctx{ctx} · V{V} subword · frozenPrefix {fp}/{dm} {(codecOnly ? "(CODEC-ONLY: zero learned tail)" : "(learned tail)")} · {cpu.ParamCount / 1000.0:F0}k params · {batches} batches x256 · BabyLM {train.Length:N0} toks");
        Console.WriteLine($"  uniform baseline = log2(V)/charsPerTok = {Math.Log2(V) / (_text.Length / (double)toks.Length):F2} bits/char\n");

        const int bs = 256;
        var rng = new Random(7);
        var sw = Stopwatch.StartNew();
        for (var b = 1; b <= batches; b++)
        {
            var batch = new List<(int[], int)>(bs);
            for (var i = 0; i < bs; i++) { var p = ctx + rng.Next(train.Length - ctx - 1); batch.Add((train[(p - ctx)..p], train[p])); }
            var lr = 3e-3 * Math.Min(1.0, 8.0 / L) * (1.0 - 0.9 * (b - 1) / Math.Max(1, batches - 1));   // depth-scaled: deep stacks need a lower LR or they diverge
            gt.TrainBatch(batch, lr);
            if (b % Math.Max(1, batches / 10) == 0 || b == batches)
            {
                var (bpc, acc) = EvalHeld(cpu, vocab, held, ctx, 800);
                Console.WriteLine($"  b{b,5}/{batches}  bits/char {bpc,5:F2}  top1 {acc,4:P0}  lr {lr:E1}  ({sw.Elapsed.TotalSeconds,4:F0}s)");
                Console.WriteLine($"      \"{Generate(cpu, vocab)}\"");
            }
        }

        var counts = new Dictionary<int, int>();
        foreach (var t in train) counts[t] = counts.GetValueOrDefault(t) + 1;
        DriftReport(cpu, vocab, dm, counts);

        Console.WriteLine("\ndone. Falling bits/char + samples that drift toward real words/grammar = it's learning to talk. Then we sweep sizes.");
        GpuDevice.Shutdown();
    }

    // DEPTH × FREEZE grid: does going DEEP close the codec-only gap? For each depth L, train codec-only (tail 0) vs a
    // tiny tail (2) vs the big learned tail (64), same budget, and print the bits/char grid. If codec-only catches up (or
    // wins) as L grows, "free embedding → all budget into depth" is the real lever. Shared corpus; one model per cell.
    public static void RunDepthFreeze(int batches = 600, int ctx = 96, int dm = 128, int S = 64)
    {
        try { Console.OutputEncoding = Encoding.UTF8; } catch { }
        _text = LoadCorpus(6_000_000);
        var vocab = new SubwordVocab(SubwordTable.List.ToList());
        var V = vocab.Size;
        var toks = vocab.Encode(_text);
        var cut = (int)(toks.Length * 0.95);
        var train = toks[..cut]; var held = toks[cut..];
        double[] Seed(int w) { var f = w < V ? PhasorCodec.Encode(vocab.Symbol(w)) : new double[PhasorCodec.Dim]; if (dm >= f.Length) return f; var s = new double[dm]; Array.Copy(f, s, dm); return s; }

        var Ls = new[] { 6, 12 };         // shallow vs deep (both within the memory ceiling) → read the depth TREND
        var tails = new[] { 0, 2, 64 };   // codec-only · one learnable phasor · big learned tail
        Console.WriteLine($"DEPTH × FREEZE — does depth close the codec-only gap? — {GpuDevice.Describe}");
        Console.WriteLine($"  d{dm} S{S} ctx{ctx} · V{V} subword · {batches} batches x256 · bits/char (lower=better) · uniform {Math.Log2(V) / (_text.Length / (double)toks.Length):F2}\n");
        Console.WriteLine($"   {"L",-4}{"tail0 codec-only",18}{"tail2 phasor",14}{"tail64 learned",16}   codec-only sample");

        foreach (var L in Ls)
        {
            var row = new double[tails.Length]; string samp0 = "";
            for (var ti = 0; ti < tails.Length; ti++)
            {
                var frozen = Math.Max(0, dm - tails[ti]);
                var cpu = new AlgFormer(V, shifts: S, layers: L, maxContext: ctx, dModel: dm, frozenPrefix: frozen, embedSeed: Seed, seed: 1);
                using var gt = new GpuTrainer(cpu);
                var rng = new Random(7);
                const int bs = 256;
                for (var b = 1; b <= batches; b++)
                {
                    var batch = new List<(int[], int)>(bs);
                    for (var i = 0; i < bs; i++) { var p = ctx + rng.Next(train.Length - ctx - 1); batch.Add((train[(p - ctx)..p], train[p])); }
                    gt.TrainBatch(batch, 3e-3 * (1.0 - 0.9 * (b - 1) / Math.Max(1, batches - 1)));
                }
                row[ti] = EvalHeld(cpu, vocab, held, ctx, 800).bpc;
                if (ti == 0) { samp0 = Generate(cpu, vocab); if (samp0.Length > 60) samp0 = samp0[..60]; }
            }
            Console.WriteLine($"   L{L,-3}{row[0],18:F2}{row[1],14:F2}{row[2],16:F2}   \"{samp0}\"");
        }
        Console.WriteLine("\ndone. If the tail0 column DROPS toward (or below) tail64 as L rises, depth substitutes for the learned tail → codec-only + depth is the lever.");
        GpuDevice.Shutdown();
    }

    // TAIL-SIZE sweep: how many LEARNABLE dims per token (beyond the frozen codec face) actually matter? tail=0 is
    // codec-only; tail=2 is one learnable phasor (the "binary choice"); up to a large tail. Finds the sweet spot where a
    // tiny learned tail recovers the codec-only gap at trivial cost (tail×V params). Shared corpus, one config per tail.
    public static void RunTail(int batches = 800, int ctx = 96, int L = 6, int dm = 128, int S = 64)
    {
        try { Console.OutputEncoding = Encoding.UTF8; } catch { }
        _text = LoadCorpus(6_000_000);
        var vocab = new SubwordVocab(SubwordTable.List.ToList());
        var V = vocab.Size;
        var toks = vocab.Encode(_text);
        var cut = (int)(toks.Length * 0.95);
        var train = toks[..cut]; var held = toks[cut..];
        double[] Seed(int w) { var f = w < V ? PhasorCodec.Encode(vocab.Symbol(w)) : new double[PhasorCodec.Dim]; if (dm >= f.Length) return f; var s = new double[dm]; Array.Copy(f, s, dm); return s; }

        Console.WriteLine($"TAIL-SIZE sweep — how many LEARNABLE dims per token matter? — {GpuDevice.Describe}");
        Console.WriteLine($"  d{dm} L{L} S{S} ctx{ctx} · V{V} subword · {batches} batches x256 · tail=0 is codec-only, tail=2 is one learnable phasor");
        Console.WriteLine($"  uniform baseline = {Math.Log2(V) / (_text.Length / (double)toks.Length):F2} bits/char\n");
        Console.WriteLine($"   {"tail",-5}{"emb-learned",13}{"bits/char",11}{"top1",7}   sample");

        foreach (var tail in new[] { 0, 2, 8, 32, 64 })
        {
            var frozen = Math.Max(0, dm - tail);
            var cpu = new AlgFormer(V, shifts: S, layers: L, maxContext: ctx, dModel: dm, frozenPrefix: frozen, embedSeed: Seed, seed: 1);
            using var gt = new GpuTrainer(cpu);
            var rng = new Random(7);
            const int bs = 256;
            for (var b = 1; b <= batches; b++)
            {
                var batch = new List<(int[], int)>(bs);
                for (var i = 0; i < bs; i++) { var p = ctx + rng.Next(train.Length - ctx - 1); batch.Add((train[(p - ctx)..p], train[p])); }
                gt.TrainBatch(batch, 3e-3 * (1.0 - 0.9 * (b - 1) / Math.Max(1, batches - 1)));
            }
            var (bpc, acc) = EvalHeld(cpu, vocab, held, ctx, 800);
            var learned = (long)tail * V;
            var samp = Generate(cpu, vocab); if (samp.Length > 80) samp = samp[..80];
            Console.WriteLine($"   {tail,-5}{learned,13:N0}{bpc,11:F2}{acc,7:P0}   \"{samp}\"");
        }
        Console.WriteLine("\ndone. If bits/char drops sharply from tail 0→2→8 then flattens, a TINY learned tail is the sweet spot: near-codec-only cost, learned-tail quality.");
        GpuDevice.Shutdown();
    }

    // COMPRESSION test: per token, learned tail ‖trained − codec_seed‖ vs atom ‖seed‖. ratio&lt;1 = the token cost less to
    // learn than its own deterministic face carries = compressed. Bucketed by token length (specialisation) + corpus-weighted.
    static void DriftReport(AlgFormer cpu, SubwordVocab vocab, int dm, Dictionary<int, int> counts)
    {
        double[] Seed(int w) { var f = PhasorCodec.Encode(vocab.Symbol(w)); if (dm >= f.Length) return f; var s = new double[dm]; Array.Copy(f, s, dm); return s; }
        double Nrm(double[] a, double[]? b) { double s = 0; for (var i = 0; i < dm; i++) { var d = b == null ? a[i] : a[i] - b[i]; s += d * d; } return Math.Sqrt(s); }

        // buckets 0..3 = token length 1,2,3,4+
        var n = new int[4]; var sumR = new double[4]; var sumTail = new double[4]; var sumAtom = new double[4];
        var wFreq = new double[4]; var wRatio = new double[4]; var sumFreq = new double[4];
        double totTail2 = 0, totAtom2 = 0; int compressed = 0; double compFreq = 0, totFreq = 0;
        for (var w = 0; w < vocab.Size; w++)
        {
            var sym = vocab.Symbol(w); var len = Math.Max(1, sym.Length); var bkt = Math.Min(len, 4) - 1;
            var seed = Seed(w); var trained = cpu.EmbRow(w);
            var atom = Nrm(seed, null); if (atom < 1e-9) continue;
            var tail = Nrm(trained, seed); var ratio = tail / atom;
            var f = counts.GetValueOrDefault(w);
            n[bkt]++; sumR[bkt] += ratio; sumTail[bkt] += tail; sumAtom[bkt] += atom; sumFreq[bkt] += f;
            wFreq[bkt] += f; wRatio[bkt] += f * ratio;
            totTail2 += tail * tail; totAtom2 += atom * atom;
            if (ratio < 0.5) { compressed++; compFreq += f; }
            totFreq += f;
        }
        Console.WriteLine("\n── COMPRESSION: learned tail ‖trained−seed‖ ÷ atom ‖seed‖  (ratio<1 ⇒ token cost less to learn than its face carries) ──");
        Console.WriteLine($"   {"len",-4}{"tokens",8}{"avg tail/atom",15}{"corpus-wtd",12}{"avg freq",10}   read");
        var tag = new[] { "chars: polysemous, paid in DEPTH", "", "", "specialised: near-frozen, COMPRESSED" };
        for (var b = 0; b < 4; b++)
        {
            if (n[b] == 0) continue;
            var lbl = b == 3 ? "4+" : (b + 1).ToString();
            Console.WriteLine($"   {lbl,-4}{n[b],8}{sumR[b] / n[b],15:F2}{(wFreq[b] > 0 ? wRatio[b] / wFreq[b] : 0),12:F2}{sumFreq[b] / n[b],10:F0}   {tag[b]}");
        }
        Console.WriteLine($"   overall learned/atom energy = {Math.Sqrt(totTail2 / Math.Max(1e-9, totAtom2)),5:F2}   ·   tokens with ratio<0.5: {compressed}/{vocab.Size} ({compressed / (double)vocab.Size,4:P0}) unweighted, {(totFreq > 0 ? compFreq / totFreq : 0),4:P0} of corpus traffic");
        Console.WriteLine("   → if short/frequent tokens show HIGH ratio and long tokens LOW, the specialisation-compression bet holds: big vocab rides frozen faces for free.");
    }

    static string LoadCorpus(int maxChars)
    {
        var dir = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Prism", "data", "text");
        var files = Directory.Exists(dir) ? Directory.GetFiles(dir, "babylm-*.txt").OrderBy(f => f).ToArray() : Array.Empty<string>();
        var sb = new StringBuilder(maxChars);
        foreach (var f in files) { if (sb.Length >= maxChars) break; sb.Append(Norm(File.ReadAllText(f))); sb.Append(' '); }
        if (sb.Length == 0) throw new InvalidOperationException($"no BabyLM corpus found under {dir}");
        return sb.Length > maxChars ? sb.ToString(0, maxChars) : sb.ToString();
    }

    // fold to printable ASCII 32..126 (the vocab range); newlines/controls → space
    static string Norm(string s)
    {
        var sb = new StringBuilder(s.Length);
        foreach (var ch in s) sb.Append(ch >= 32 && ch <= 126 ? ch : ' ');
        return sb.ToString();
    }

    static (double bpc, double acc, long prm, double charsPerTok, string curve, string sample) TrainEval(
        SubwordVocab vocab, int L, int dm, int c, int S, int batches)
    {
        var V = vocab.Size;
        var toks = vocab.Encode(_text);
        var cut = (int)(toks.Length * 0.95);
        var train = toks[..cut]; var held = toks[cut..];
        var charsPerTok = _text.Length / (double)toks.Length;

        double[] Seed(int w)
        {
            var f = w < V ? PhasorCodec.Encode(vocab.Symbol(w)) : new double[PhasorCodec.Dim];
            if (dm >= f.Length) return f;
            var s = new double[dm]; Array.Copy(f, s, dm); return s;
        }
        var frozen = Math.Min(PhasorCodec.FrozenReals, dm);
        var cpu = new AlgFormer(V, shifts: S, layers: L, maxContext: c, dModel: dm, frozenPrefix: frozen, embedSeed: Seed, seed: 1);
        using var gt = new GpuTrainer(cpu);
        var rng = new Random(7);

        const int bs = 128;
        var curve = new List<string>();
        for (var b = 1; b <= batches; b++)
        {
            var batch = new List<(int[], int)>(bs);
            for (var i = 0; i < bs; i++) { var p = c + rng.Next(train.Length - c - 1); batch.Add((train[(p - c)..p], train[p])); }
            var lr = 2e-3 * (1.0 - 0.85 * (b - 1) / Math.Max(1, batches - 1));
            gt.TrainBatch(batch, lr);
            if (b % Math.Max(1, batches / 4) == 0) { var (bpc0, _) = EvalHeld(cpu, vocab, held, c, 300); curve.Add(bpc0.ToString("F2")); }
        }
        var (bpc, acc) = EvalHeld(cpu, vocab, held, c, 800);
        return (bpc, acc, cpu.ParamCount, charsPerTok, string.Join("→", curve), Generate(cpu, vocab));
    }

    // bits-per-CHARACTER: sum token cross-entropy (bits) and divide by the chars those tokens spell → tokenizer-fair
    static (double bpc, double acc) EvalHeld(AlgFormer m, SubwordVocab vocab, int[] held, int c, int n)
    {
        var r = new Random(123);
        double bits = 0, chars = 0; int ok = 0, cnt = 0;
        for (var i = 0; i < n; i++)
        {
            var p = c + r.Next(held.Length - c - 1);
            var ctx = held[(p - c)..p]; var tgt = held[p];
            var lg = m.LogitsFor(ctx);
            var max = double.NegativeInfinity; var arg = 0;
            for (var w = 0; w < lg.Length; w++) { if (lg[w] > max) { max = lg[w]; arg = w; } }
            double sum = 0; for (var w = 0; w < lg.Length; w++) sum += Math.Exp(lg[w] - max);
            var pt = Math.Exp(lg[tgt] - max) / sum;
            bits += -Math.Log2(Math.Max(pt, 1e-12));
            chars += Math.Max(1, vocab.Symbol(tgt).Length);
            if (arg == tgt) ok++;
            cnt++;
        }
        return (bits / chars, ok / (double)cnt);
    }

    static string Generate(AlgFormer m, SubwordVocab vocab)
    {
        var prompt = vocab.Encode("The ");
        var outIds = m.Generate(prompt, 120, temperature: 0.8, seed: 1);
        var text = vocab.Decode(outIds).Replace('\n', ' ');
        return text.Length > 160 ? text[..160] : text;
    }
}
