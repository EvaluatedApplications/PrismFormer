// Copyright (c) 2026 Dongyang Stephen Chen, trading as Evaluated Applications. All rights reserved.
// Source-available for study and verification; commercial use, redistribution, or derivatives require permission. See LICENSE.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using PrismFormer;

namespace PrismFormer.Bench;

/// <summary>
/// --lrsweep : WHAT LEARNING RATE ACTUALLY WORKS for THIS model shape on real corpus next-token? The live model trains at a
/// deliberately low ~1.9e-4 (depth-scaled, chosen for mesh consensus + anti-poison stability), and it plateaus around 9-10
/// nats — the open question is whether that rate is the throttle. This sweeps a FRESH, full production model
/// (PrismSpec.NewModel = L32 / d256 / S64, frozen codec, identity/ReZero init — the real shape, NOT a shrunk toy) at each of a
/// range of LRs from the current rate up to a genuinely RIDICULOUS 3.0, on a FIXED slice of the real corpus, and reports the
/// next-token loss (mean CE, nats) each LR reaches. Same init, same sampled examples, same shuffle for every LR → the only
/// variable is the rate. This is the Leslie-Smith LR range test: the steepest-descending stable LR is the bootstrap rate; the
/// point where it explodes is the ceiling. Because the frozen-codec + ReZero regime is unexplored, the usual "you diverge past
/// 1e-3" transformer ceiling is only an assumption here — we map where it REALLY breaks. Divergent LRs early-exit so the
/// reckless end is cheap. Windows are capped short (WinCtx) purely so the deep stack is tractable in a bench; the MODEL is full
/// ctx-512 and the LR verdict transfers (optimizer dynamics track the model dims, not the training sequence length).
/// </summary>
public static class LrSweepBench
{
    public static void Run(int epochs = 5)
    {
        try { Console.OutputEncoding = System.Text.Encoding.UTF8; } catch { }

        const int WinCtx = 96;    // training-window length — short so L32 is tractable here; the model itself is full ctx-512
        const int N = 160;        // sampled next-token examples — one FIXED slice, identical for every LR
        const int Batch = 16;
        var par = Math.Max(2, Environment.ProcessorCount / 2);   // POLITE: leave half the cores for live Studio training
        // From the current live rate up into genuinely ridiculous territory. Adam normalises the gradient scale, so lr≈the
        // per-step move; lr=1.0/3.0 would obliterate a normal net — the question is whether frozen-codec + ReZero survives it.
        var lrs = new[] { 1.9e-4, 3.75e-4, 7.5e-4, 1.5e-3, 3e-3, 6e-3, 1.2e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0 };

        var v = new CharVocab();
        double[] Seed(int w) => PhasorCodec.Encode(v.Symbol(w));   // frozen codec seed — EXACTLY the live StudioModel.Seed

        // Prefer the LIVE corpus the user is actually training on; fall back to the repo's shipped text.
        var live = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Prism", "data", "text");
        var dir = HasText(live) ? live : FindRepoText();
        if (!HasText(dir)) { Console.WriteLine("no corpus .txt found (looked in " + live + " and the repo studio/PrismStudio/data/text)"); return; }

        Console.WriteLine($"loading corpus from {dir} …");
        var corpus = CorpusSource.FromFolders(WinCtx, v, dir);
        if (corpus.Count <= 0) { Console.WriteLine("corpus produced 0 windows (all segments shorter than the window?)"); return; }

        // One fixed sample, drawn once, reused for every LR (seeded → reproducible, fair across rates).
        var rng = new Random(12345);
        var sample = new List<(int[] Ctx, int Target)>(N);
        for (var i = 0; i < N; i++) sample.Add(corpus.GetExample((long)(rng.NextDouble() * corpus.Count)));

        var rand = Math.Log(PrismSpec.Vocab);   // random-guess CE (nats) — the "learned nothing" line
        Console.WriteLine($"\nLR SWEEP — full production shape {PrismSpec.Signature}");
        Console.WriteLine($"  fresh identity-init model per LR · {N} corpus windows (ctx {WinCtx}) · batch {Batch} · {epochs} epochs (~{Math.Max(1, N / Batch) * epochs} Adam steps each) · {par} cores (half left for live training)");
        Console.WriteLine($"  next-token loss = mean CE in nats · random-guess = ln({PrismSpec.Vocab}) = {rand:F2} · 'talking' ~7");
        Console.WriteLine($"  NB: identity-init start is ~126 (confident repeat-last-token, not random) — loss falls FROM there\n");
        Console.WriteLine($"  {"lr",10}   {"per-epoch loss (nats)",-34}{"best",8}{"drop",8}   status");
        Console.WriteLine("  " + new string('-', 74));

        double bestLoss = double.PositiveInfinity, bestLr = 0;
        foreach (var lr in lrs)
        {
            var m = PrismSpec.NewModel(Seed);   // fresh, deterministic identity-init — identical start for every rate
            var losses = new List<double>();
            var diverged = false;
            for (var ep = 1; ep <= epochs; ep++)
            {
                var l = m.TrainEpoch(sample, Batch, lr, shuffleSeed: 7, parallelism: par);
                losses.Add(l);
                // The identity-init start is ~126 nats (confident repeat-last-token), so loss falls FROM there — divergence is
                // NaN/Inf, a hard blow-up, or loss RISING well above where this rate started (not "above random").
                if (double.IsNaN(l) || double.IsInfinity(l) || l > 1e3 || (losses.Count >= 2 && l > losses[0] * 1.3)) { diverged = true; break; }
            }
            var best = losses.Count == 0 ? double.NaN : losses.Min();
            var drop = losses.Count == 0 ? 0 : losses[0] - best;   // how far it fell from its own start = descent in this budget
            var trace = string.Join(" ", losses.Select(x => x.ToString("F1")));
            var status = diverged ? "DIVERGED" : (drop > 2 ? "descending" : "flat");
            Console.WriteLine($"  {lr,10:0.####e-0}   {trace,-34}{best,8:F2}{drop,8:F2}   {status}");
            if (!diverged && best < bestLoss) { bestLoss = best; bestLr = lr; }
        }

        Console.WriteLine("  " + new string('-', 74));
        if (bestLr > 0)
        {
            Console.WriteLine($"\n  best stable rate: lr={bestLr:0.####e-0} → {bestLoss:F2} nats (live rate is 1.9e-4).");
            Console.WriteLine("  read: 'drop' = how far it fell from epoch 1 = descent speed. The highest lr that still says 'learning'");
            Console.WriteLine("  (not DIVERGED) with a big drop is the bootstrap rate; the first DIVERGED lr is the ceiling for this shape.");
            Console.WriteLine("  NB short budget — this ranks INITIAL descent, not final converged loss; the ordering is the signal.");
        }
        else Console.WriteLine("\n  every rate diverged or stayed flat — unexpected; check the corpus/sample.");
    }

    static bool HasText(string dir) => !string.IsNullOrEmpty(dir) && Directory.Exists(dir)
        && Directory.EnumerateFiles(dir, "*.txt", SearchOption.AllDirectories).Any();

    static string FindRepoText()
    {
        var d = AppContext.BaseDirectory;
        for (var i = 0; i < 10 && !string.IsNullOrEmpty(d); i++)
        {
            var t = Path.Combine(d, "studio", "PrismStudio", "data", "text");
            if (Directory.Exists(t)) return t;
            d = Directory.GetParent(d)?.FullName;
        }
        return "";
    }
}
