// Copyright (c) 2026 Dongyang Stephen Chen, trading as Evaluated Applications. All rights reserved.
// Source-available for study and verification; commercial use, redistribution, or derivatives require permission. See LICENSE.

using PrismFormer;

namespace PrismFormer.Bench;

/// <summary>
/// ATOMIC CAPACITY (--capacity). A raw associative-memory showdown: an equal-sized transformer and PrismFormer
/// (AlgFormer) compete to MEMORISE the most unique atomic facts. We build a key→value STORE with NO structure to
/// generalise from — each key is a short unique token-sequence over a fixed alphabet and its value is an ARBITRARY
/// (uniform-random) value token, so nothing can be inferred; every fact must be STORED. We sweep the number of
/// facts N upward, train BOTH models to memorise all N (equal exposure per fact at every N, so recall fall-off is
/// pure storage capacity, not budget starvation), then measure recall on those same N facts (this is memorisation,
/// not held-out — capacity, not generalisation). We report the recall-vs-N curve for both and the "atomic capacity"
/// = the largest N at ≥95% and at ≥50% recall.
///
/// <para>Facts reuse the benches' tokenisation convention: a key is <see cref="KeyLen"/> symbols drawn from a
/// <see cref="KeyAlphabet"/>-symbol alphabet (so 16^4 = 65536 unique keys are available), followed by "="; the
/// target is one of <see cref="ValueSet"/> value tokens. The alphabet is FIXED, so vocab and therefore the models'
/// parameter budget do NOT grow with N — a clean fixed-model capacity measure. Chance recall = 1/ValueSet.</para>
///
/// <para>Parameter-matched at the repo's production convention: AlgFormer at the head-to-head config (dModel=256,
/// shifts=8, layers=2, frozen identity), the transformer's (d, dff, L) auto-searched to match its ParamCount, using
/// the MODERN pre-norm LayerNorm recipe (warmup→cosine + tuned Adam) — the strongest fair baseline for a pure
/// memorisation contest. Same corpus, train order, and exposure for both.</para>
/// </summary>
public static class CapacityBench
{
    const int KeyAlphabet = 16;   // key symbols s0..s15 (hashed phasor signatures — no numeric structure)
    const int KeyLen = 4;         // key = 4 symbols → 16^4 = 65536 unique keys available
    const int ValueSet = 256;     // value tokens v0..v255; arbitrary key→value assignment; chance recall = 1/256

    public static void Run(int maxN = 1024, int passes = 400, bool tuned = true)
    {
        try { Console.OutputEncoding = System.Text.Encoding.UTF8; } catch { }

        // ---- fixed vocab (built up-front so tokens/seeds are stable across every N) ----
        var id = new Dictionary<string, int>(StringComparer.Ordinal) { ["<pad>"] = 0 };
        var words = new List<string> { "<pad>" };
        int Id(string w) { if (id.TryGetValue(w, out var i)) return i; i = id.Count; id[w] = i; words.Add(w); return i; }
        var eq = Id("=");
        var keySym = new int[KeyAlphabet]; for (var i = 0; i < KeyAlphabet; i++) keySym[i] = Id($"s{i}");
        var valSym = new int[ValueSet]; for (var i = 0; i < ValueSet; i++) valSym[i] = Id($"v{i}");
        var vocab = Math.Max(1024, id.Count + 8);
        double[] Seed(int w) => w < words.Count ? PhasorCodec.Encode(words[w]) : new double[PhasorCodec.Dim];

        // ---- models at the repo's production head-to-head convention; transformer auto-matched to AlgFormer params ----
        const int ctx = 16;
        var probe = new AlgFormer(vocab, shifts: 8, layers: 2, maxContext: ctx, dModel: PhasorCodec.Dim, frozenPrefix: PhasorCodec.FrozenReals, embedSeed: Seed, seed: 1);
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
        Console.WriteLine($"params: PrismFormer {targetParams:N0} (d={PhasorCodec.Dim}, S=8, L=2)   transformer {xfParams:N0} (d={best.d}, dff={best.ff}, L={best.L}{(tuned ? ", +LayerNorm" : "")})   ({(double)xfParams / targetParams:F2}x pound-for-pound)");
        Console.WriteLine($"key = {KeyLen} symbols over a {KeyAlphabet}-symbol alphabet + \"=\"  ·  value = 1 of {ValueSet} tokens (arbitrary, must be stored)  ·  chance recall {1.0 / ValueSet:P1}");
        Console.WriteLine($"train BOTH to memorise all N facts ({passes} passes each, equal exposure per fact at every N)  ·  vocab {id.Count}\n");

        // ---- build N arbitrary facts (identical for both models); keys unique, values uniform-random ----
        (int[] key, int val)[] MakeFacts(int N)
        {
            var r = new Random(12345);   // fixed → both models memorise the exact same store
            var seen = new HashSet<string>();
            var facts = new List<(int[], int)>(N);
            while (facts.Count < N)
            {
                var ks = new int[KeyLen];
                for (var k = 0; k < KeyLen; k++) ks[k] = r.Next(KeyAlphabet);
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

        // tuned transformer LR: warmup→cosine over passes; AlgFormer keeps the shared linear-decay base (its own recipe).
        var warm = Math.Max(10, passes / 20);
        double LrTuned(int ep) { const double peak = 2e-3; if (ep <= warm) return peak * ep / warm; var t = (ep - warm) / (double)Math.Max(1, passes - warm); return peak * (0.05 + 0.95 * 0.5 * (1 + Math.Cos(Math.PI * t))); }

        var Ns = new List<int>(); for (var n = 32; n <= maxN; n *= 2) Ns.Add(n);
        var curveP = new List<(int N, double r)>(); var curveX = new List<(int N, double r)>();

        Console.WriteLine("recall on the stored facts (memorisation) --------------------------------------");
        Console.WriteLine($"  {"N facts",8}  {"PrismFormer",12}  {"transformer",12}");
        foreach (var N in Ns)
        {
            var facts = MakeFacts(N);
            var alg = new AlgFormer(vocab, shifts: 8, layers: 2, maxContext: ctx, dModel: PhasorCodec.Dim, frozenPrefix: PhasorCodec.FrozenReals, embedSeed: Seed, seed: 1);
            var xf = new MiniTransformer(vocab, best.d, best.ff, best.L, ctx, 42, layerNorm: tuned);

            // train both identically (same shuffled order each pass); independent models → run side by side
            void Train(Action<int[], int, double, double> stepAlg, Action<int[], int, double, double> stepXf)
            {
                for (var ep = 1; ep <= passes; ep++)
                {
                    var lr = 2e-3 * (1.0 - 0.9 * (ep - 1) / Math.Max(1, passes - 1));   // shared base schedule
                    var lx = tuned ? LrTuned(ep) : lr;                                  // transformer: tuned warmup→cosine when on
                    var order = Enumerable.Range(0, facts.Length).ToArray();
                    var er = new Random(1000 + ep);
                    for (var i = order.Length - 1; i > 0; i--) { var j = er.Next(i + 1); (order[i], order[j]) = (order[j], order[i]); }
                    Parallel.Invoke(
                        () => { foreach (var idx in order) { var (key, val) = facts[idx]; stepAlg(key, val, lr, ep); } },
                        () => { foreach (var idx in order) { var (key, val) = facts[idx]; stepXf(key, val, lx, ep); } });
                }
            }
            Train(
                (key, val, lr, _) => alg.TrainStep(key, val, lr),
                (key, val, lx, _) => { if (tuned) xf.TrainStep(key, val, lx, 0.9, 0.999, 1e-8); else xf.TrainStep(key, val, lx); });

            var rp = Recall(alg.Predict, facts); var rx = Recall(xf.Predict, facts);
            curveP.Add((N, rp)); curveX.Add((N, rx));
            Console.WriteLine($"  {N,8}  {rp,12:P1}  {rx,12:P1}");
        }

        int Capacity(List<(int N, double r)> c, double thr) { var cap = 0; foreach (var (N, r) in c) if (r >= thr) cap = N; return cap; }
        Console.WriteLine("\natomic capacity (largest N meeting the recall bar) -----------------------------");
        Console.WriteLine($"  ≥95% recall :  PrismFormer {Capacity(curveP, 0.95),6:N0}     transformer {Capacity(curveX, 0.95),6:N0}");
        Console.WriteLine($"  ≥50% recall :  PrismFormer {Capacity(curveP, 0.50),6:N0}     transformer {Capacity(curveX, 0.50),6:N0}");
        Console.WriteLine("\nmemorisation, not held-out: every fact is arbitrary (no structure), so recall = raw associative-memory capacity.");
    }
}
