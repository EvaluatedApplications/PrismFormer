// Copyright (c) 2026 Dongyang Stephen Chen, trading as Evaluated Applications. All rights reserved.
// Source-available for study and verification; commercial use, redistribution, or derivatives require permission. See LICENSE.

using PrismFormer;

namespace PrismFormer.Bench;

/// <summary>
/// ATOMIC CAPACITY (--capacity). A raw associative-memory showdown: an equal-sized transformer and PrismFormer
/// (AlgFormer) compete to MEMORISE the most unique atomic facts. We build a key→value STORE with NO structure to
/// generalise from — each key is a short unique token-sequence over a fixed alphabet and its value is an ARBITRARY
/// (uniform-random) value token, so nothing can be inferred; every fact must be STORED. We sweep the number of
/// facts N upward and train BOTH models with EARLY-STOP — each runs until it converges to full recall (it fits N) or
/// plateaus (N exceeds its capacity), which is the truer capacity test than a fixed budget (a fixed budget conflates
/// how FAST a model learns with how MUCH it can hold). We measure recall on those same N facts (memorisation, not
/// held-out) and report the recall-vs-N curve plus the "atomic capacity" = the largest N still at ≥95% / ≥50% recall.
///
/// <para>Each key is <see cref="KeyLen"/> atomic symbols over a small alphabet sized so its tuples cover maxN unique
/// keys, followed by "="; the target is one of <see cref="ValueSet"/> random value tokens. The alphabet stays small and
/// FIXED, so vocab (hence both models' parameter budget) does NOT grow with N. A single-token key would hand each key
/// its own embedding (free storage → no real ceiling); shared symbols force the fixed machinery to STORE the map. Every
/// (N, model) trains as an independent job, all run CONCURRENTLY. Chance recall = 1/ValueSet.</para>
///
/// <para>Parameter-matched at the repo's production convention: AlgFormer at the head-to-head config (dModel=256,
/// shifts=8, layers=2, frozen identity), the transformer's (d, dff, L) auto-searched to match its ParamCount, using
/// the MODERN pre-norm LayerNorm recipe (warmup→cosine + tuned Adam) — the strongest fair baseline for a pure
/// memorisation contest. Same corpus, train order, and exposure for both.</para>
/// </summary>
public static class CapacityBench
{
    const int KeyLen = 2;         // MINIMAL keys: 2 atomic symbols. Short, but drawn from a small FIXED alphabet so the
                                  // vocab (hence both models' parameter budget) stays ~constant as N grows. A single-token
                                  // key would hand each key its OWN learnable embedding, so the model would grow with N and
                                  // never saturate (capacity would just track the vocab). Shared symbols force the fixed
                                  // machinery to STORE the map — which is what makes the capacity ceiling real.
    const int ValueSet = 256;     // value = 1 of 256 random tokens (v0..v255); arbitrary key→value; chance recall = 1/256
    const int KeyAlphabet = 128;  // FIXED, independent of maxN. RIGOUR: vocab (hence both models' param budget) stays IDENTICAL at every N and across every run, so the recall-vs-N curve is comparable run-to-run. 128^KeyLen = 16,384 distinct keys covers maxN into the low tens of thousands. (Was √maxN, which drifted the model size between runs.)

    public static void Run(int maxN = 2048, int passes = 400, bool tuned = true, int dModel = 128)
    {   // dModel < PhasorCodec.Dim keeps the frozen codec identity + a learned tail (a genuine, smaller PrismFormer). AlgFormer supports it; the probe, Seed, and the training job below must all use the SAME width (dm/frozen).
        try { Console.OutputEncoding = System.Text.Encoding.UTF8; } catch { }

        // ---- fixed vocab (built up-front so tokens/seeds are stable across every N) ----
        var id = new Dictionary<string, int>(StringComparer.Ordinal) { ["<pad>"] = 0 };
        var words = new List<string> { "<pad>" };
        int Id(string w) { if (id.TryGetValue(w, out var i)) return i; i = id.Count; id[w] = i; words.Add(w); return i; }
        var eq = Id("=");
        var maxKeys = (int)Math.Pow(KeyAlphabet, KeyLen);
        if (maxN > (int)(0.85 * maxKeys)) maxN = (int)(0.85 * maxKeys);   // keep unique-key rejection sampling fast (never crowd the key space); raise KeyAlphabet for a larger sweep
        var keyAlphabet = KeyAlphabet;   // FIXED (not √maxN) → constant vocab/model size across every N and run
        var keySym = new int[keyAlphabet]; for (var i = 0; i < keyAlphabet; i++) keySym[i] = Id($"s{i}");
        var valSym = new int[ValueSet]; for (var i = 0; i < ValueSet; i++) valSym[i] = Id($"v{i}");
        var vocab = id.Count + 8;   // no 1024 floor: the pad rows never train and just bloat the embedding (88% of params) and the per-token softmax cost — dead weight, not capacity
        var dm = Math.Min(dModel, PhasorCodec.Dim);         // model width. A SMALLER PrismFormer of the SAME TYPE: dm>=64 keeps the full frozen codec identity + a learned tail (still a real PrismFormer, just a smaller container). Only dm<64 would drop into codec-less orbital-only.
        var frozen = Math.Min(PhasorCodec.FrozenReals, dm);
        double[] Seed(int w)
        {
            var full = w < words.Count ? PhasorCodec.Encode(words[w]) : new double[PhasorCodec.Dim];
            if (dm >= full.Length) return full;
            var s = new double[dm]; Array.Copy(full, s, dm); return s;   // truncate the codec vector to the model width (keeps the frozen identity prefix intact)
        }

        // ---- models: a SMALL PrismFormer (codec intact) vs a size-matched small transformer — smallest valid container of each type, so the run is minutes ----
        const int ctx = 16;
        var probe = new AlgFormer(vocab, shifts: 8, layers: 2, maxContext: ctx, dModel: dm, frozenPrefix: frozen, embedSeed: Seed, seed: 1);
        var targetParams = probe.ParamCount;
        (int d, int L, int ff) best = (32, 2, 64); var bestDelta = long.MaxValue;
        foreach (var d in new[] { 24, 32, 40, 48, 56, 64, 96, 128, 160, 192, 256 })
            foreach (var L in new[] { 2, 3, 4 })
                foreach (var ff in new[] { 32, 48, 64, 96, 128, 160, 192, 256, 384, 512, 768, 1024 })
                { var pr = new MiniTransformer(vocab, d, ff, L, ctx, 42, layerNorm: tuned); var delta = Math.Abs(pr.ParamCount - targetParams); if (delta < bestDelta) { bestDelta = delta; best = (d, L, ff); } }
        var xfParams = new MiniTransformer(vocab, best.d, best.ff, best.L, ctx, 42, layerNorm: tuned).ParamCount;

        Console.WriteLine($"PrismFormer vs pound-for-pound transformer — ATOMIC CAPACITY (unique-fact memorisation)   {DateTime.Now:yyyy-MM-dd HH:mm}");
        if (!AlgFormer.GradCheck(out var ar) || !MiniTransformer.GradCheck(out var xr, layerNorm: tuned)) { Console.WriteLine("GRADCHECK FAILED — aborting"); return; }
        Console.WriteLine($"gradchecks pass (prism {ar:E1}, transformer{(tuned ? " +LN" : "")} {xr:E1})");
        Console.WriteLine($"params: PrismFormer {targetParams:N0} (d={dm}, S=8, L=2, frozen={frozen})   transformer {xfParams:N0} (d={best.d}, dff={best.ff}, L={best.L}{(tuned ? ", +LayerNorm" : "")})   ({(double)xfParams / targetParams:F2}x pound-for-pound)");
        Console.WriteLine($"key = {KeyLen} symbols over a {keyAlphabet}-symbol alphabet + \"=\"  ·  value = 1 of {ValueSet} tokens (arbitrary, must be stored)  ·  chance recall {1.0 / ValueSet:P1}  ·  vocab {id.Count} (FIXED as N grows)");
        Console.WriteLine($"train BOTH with early-stop (cap {passes} passes; stop at convergence or plateau)  ·  concurrent jobs  ·  vocab {id.Count}\n");

        // ---- build N arbitrary facts (identical for both models); keys unique, values uniform-random ----
        (int[] key, int val)[] MakeFacts(int N)
        {
            var r = new Random(12345);   // fixed → both models memorise the exact same store
            var seen = new HashSet<string>();
            var facts = new List<(int[], int)>(N);
            while (facts.Count < N)
            {
                var ks = new int[KeyLen];
                for (var k = 0; k < KeyLen; k++) ks[k] = r.Next(keyAlphabet);
                if (!seen.Add(string.Join(',', ks))) continue;   // distinct key sequence
                var key = new int[KeyLen + 1];
                for (var k = 0; k < KeyLen; k++) key[k] = keySym[ks[k]];
                key[KeyLen] = eq;
                facts.Add((key, valSym[r.Next(ValueSet)]));       // arbitrary value → nothing to infer, must store
            }
            return facts.ToArray();
        }

        double Recall(Func<int[], int> predict, (int[] key, int val)[] facts)
        {
            var ok = 0; foreach (var (key, val) in facts) if (predict(key) == val) ok++;
            return ok / (double)facts.Length;
        }

        // LR schedules run over the pass CAP; an early-stopped job just uses the early part of the curve.
        var warm = Math.Max(10, passes / 20);
        double LrTuned(int ep) { const double peak = 2e-3; if (ep <= warm) return peak * ep / warm; var t = (ep - warm) / (double)Math.Max(1, passes - warm); return peak * (0.05 + 0.95 * 0.5 * (1 + Math.Cos(Math.PI * t))); }
        double LrBase(int ep) => 2e-3 * (1.0 - 0.9 * (ep - 1) / Math.Max(1, passes - 1));

        // Coarse low anchors (flat, both-memorise region) + DENSE linear steps (512) through the break zone so the recall-vs-N
        // curve resolves where each model crosses 95%. Wall-time is set by the single slowest job, so extra points are ~free.
        var Ns = new List<int>();
        for (var n = 256; n <= 1024 && n <= maxN; n *= 2) Ns.Add(n);        // 256,512,1024 — context
        for (var n = 2048; n <= maxN; n += 512) Ns.Add(n);                  // dense through both break points
        if (Ns.Count == 0) Ns.Add(maxN); else if (Ns[^1] != maxN) Ns.Add(maxN);

        // EARLY-STOP per model: eval recall every CHECK passes; STOP when it converges (>=DONE ⇒ it fits N) or plateaus
        // (rises slower than MINGAIN/check for PLATEAU checks ⇒ N exceeds its capacity). No fixed grind. This is MORE
        // rigorous than a fixed budget: capacity = "can it converge to full recall", not "recall at K passes" (which
        // conflates speed with capacity). NB: memorisation recall creeps up MONOTONICALLY, so the plateau test must be
        // a MINIMUM SLOPE (MINGAIN), not "any gain" — a near-zero-tolerance test never fires and every job runs to cap.
        const int CHECK = 25; const double DONE = 0.98; const int PLATEAU = 3; const double MINGAIN = 0.004;
        (double r, int ep, bool conv) TrainEarly(Func<int[], int> predict, Action<int[], int, int> step, (int[] key, int val)[] facts)
        {
            double best = 0; int noGain = 0, used = 0;
            for (var ep = 1; ep <= passes; ep++)
            {
                var order = Enumerable.Range(0, facts.Length).ToArray();
                var er = new Random(1000 + ep);
                for (var i = order.Length - 1; i > 0; i--) { var j = er.Next(i + 1); (order[i], order[j]) = (order[j], order[i]); }
                foreach (var idx in order) { var (key, val) = facts[idx]; step(key, val, ep); }
                used = ep;
                if (ep % CHECK == 0 || ep == passes)
                {
                    var r = Recall(predict, facts);
                    if (r >= DONE) return (r, ep, true);
                    if (r > best + MINGAIN) { best = r; noGain = 0; } else if (++noGain >= PLATEAU) return (Math.Max(best, r), ep, false);
                    best = Math.Max(best, r);
                }
            }
            return (best, used, false);
        }

        // ONE job per (N, model-kind); ALL run concurrently to fill the cores (was 2-way Parallel.Invoke, sequential N).
        var jobs = new List<(int N, int kind)>();               // kind 0 = PrismFormer, 1 = transformer
        foreach (var N in Ns) { jobs.Add((N, 0)); jobs.Add((N, 1)); }
        jobs = jobs.OrderByDescending(j => j.N).ToList();        // longest-processing-time-first: start the big-N jobs first so they aren't end-of-run stragglers (minimises makespan)
        var res = new System.Collections.Concurrent.ConcurrentDictionary<(int, int), (double r, int ep, bool conv)>();
        Console.WriteLine($"training {jobs.Count} jobs concurrently ({Ns.Count} N × 2 models), early-stop at convergence/plateau (cap {passes} passes)…\n");
        Parallel.ForEach(System.Collections.Concurrent.Partitioner.Create(jobs, System.Collections.Concurrent.EnumerablePartitionerOptions.NoBuffering), job =>
        {
            var facts = MakeFacts(job.N);
            if (job.kind == 0)
            {
                var alg = new AlgFormer(vocab, shifts: 8, layers: 2, maxContext: ctx, dModel: dm, frozenPrefix: frozen, embedSeed: Seed, seed: 1);   // MUST match the probe + Seed width (dm/frozen); hardcoding PhasorCodec.Dim here while Seed truncates to dm gave a 256-wide model with 128-wide embeddings → the ForwardAll crash
                res[job] = TrainEarly(alg.Predict, (k, v, ep) => alg.TrainStep(k, v, LrBase(ep)), facts);
            }
            else
            {
                var xf = new MiniTransformer(vocab, best.d, best.ff, best.L, ctx, 42, layerNorm: tuned);
                res[job] = TrainEarly(xf.Predict, (k, v, ep) => { if (tuned) xf.TrainStep(k, v, LrTuned(ep), 0.9, 0.999, 1e-8); else xf.TrainStep(k, v, LrBase(ep)); }, facts);
            }
        });

        var curveP = new List<(int N, double r)>(); var curveX = new List<(int N, double r)>();
        Console.WriteLine("recall on the stored facts (early-stopped)  ·  conv=fits N, plat=exceeds capacity, @Nep=passes used --------");
        Console.WriteLine($"  {"N facts",8}  {"PrismFormer",22}  {"transformer",22}");
        string Fmt((double r, int ep, bool conv) t) => $"{t.r,7:P1} {(t.conv ? "conv" : "plat"),-4} @{t.ep,3}ep";
        foreach (var N in Ns)
        {
            var p = res[(N, 0)]; var x = res[(N, 1)];
            curveP.Add((N, p.r)); curveX.Add((N, x.r));
            Console.WriteLine($"  {N,8}  {Fmt(p),22}  {Fmt(x),22}");
        }

        int Capacity(List<(int N, double r)> c, double thr) { var cap = 0; foreach (var (N, r) in c) if (r >= thr) cap = N; return cap; }
        Console.WriteLine("\natomic capacity (largest N still memorised) ------------------------------------");
        Console.WriteLine($"  ≥95% recall :  PrismFormer {Capacity(curveP, 0.95),6:N0}     transformer {Capacity(curveX, 0.95),6:N0}");
        Console.WriteLine($"  ≥50% recall :  PrismFormer {Capacity(curveP, 0.50),6:N0}     transformer {Capacity(curveX, 0.50),6:N0}");
        if (curveP[^1].r >= 0.95 || curveX[^1].r >= 0.95) Console.WriteLine($"  NOTE: a model is still ≥95% at N={Ns[^1]} (the top) — its ceiling is HIGHER; re-run with --maxN above {maxN}.");
        Console.WriteLine("\nmemorisation, not held-out: every fact is arbitrary (no structure), so recall = raw associative-memory capacity.");
    }
}
