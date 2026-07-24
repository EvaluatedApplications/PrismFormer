// Copyright (c) 2026 Dongyang Stephen Chen, trading as Evaluated Applications. All rights reserved.
// Source-available for study and verification; commercial use, redistribution, or derivatives require permission. See LICENSE.

using PrismFormer;

namespace PrismFormer.Bench;

/// <summary>
/// --shift : DOES RANK (S) HELP THE MODEL HOLD ITS PLACE PAST THE CONTEXT WINDOW? Raw recall saturates (a model memorises
/// many short facts easily), so instead this overfits ONE sequence and free-runs it well beyond the context: prime with a
/// few chars, then generate by feeding the model its OWN output, sliding a ctx-wide window. Once generation passes ctx
/// tokens the start of the sequence has slid out of view, so to keep reproducing it correctly the model must have learned
/// the whole thing and track WHERE it is, not just pattern-match the visible window. We measure how far it stays on the rails
/// before it loses the thread (a self-fed error cascades once it slips). Sweep S: correct-run rising with S past ctx = more
/// rank helps long-range coherence; flat = tracking is limited by something else (depth / residual width), not rank.
/// </summary>
public static class ShiftBench
{
    // a LONG sequence with heavy internal recurrence (recurring phrases → the window is locally ambiguous, so the model must
    // track WHERE it is; and it is long enough that a low-rank model runs out of room to hold the whole map)
    const string TEXT = "peter piper picked a peck of pickled peppers. a peck of pickled peppers peter piper picked. if peter piper picked a peck of pickled peppers, wheres the peck that peter piper picked? she sells sea shells by the sea shore, and the shells she sells are surely sea shells. how much wood would a woodchuck chuck if a woodchuck could chuck all the wood? peter keeps picking peppers. ";

    public static void Run(int passes = 120)
    {
        try { Console.OutputEncoding = System.Text.Encoding.UTF8; } catch { }
        var v = new CharVocab();
        int d = PhasorCodec.Dim;
        const int V = CharVocab.N, L = 2, ctx = 16, prime = 8;
        double[] Seed(int w) { var f = PhasorCodec.Encode(v.Symbol(w)); if (f.Length == d) return f; var s = new double[d]; Array.Copy(f, s, Math.Min(f.Length, d)); return s; }

        var ids = TEXT.Select(c => v.Id(c)).ToArray();
        int len = ids.Length;
        Console.WriteLine($"SHIFT & LONG GENERATION — overfit ONE {len}-char sequence, free-run from {prime} chars, ctx {ctx}, d{d} L{L}");
        Console.WriteLine($"  does more rank help it stay on track once the start has slid out of the {ctx}-token window?\n");
        Console.WriteLine($"  {"S",5}{"params",12}{"memorised",11}{"correct-run",13}{"match%",9}   (memorised = teacher-forced acc; correct-run = free-run chars before it loses the thread)");

        var shifts = new[] { 2, 4, 8, 16, 32 };
        var res = new (int S, long p, double tf, int run, double match)[shifts.Length];
        Parallel.For(0, shifts.Length, si =>
        {
            var S = shifts[si];
            var m = new AlgFormer(V, shifts: S, layers: L, maxContext: ctx, dModel: d, frozenPrefix: d, embedSeed: Seed, seed: 1);   // codec-only: banks are the only learned params, scale purely with S
            for (var ep = 0; ep < passes; ep++)
            {
                var lr = 3e-3 * (1.0 - 0.85 * ep / Math.Max(1, passes - 1));
                for (var p = 1; p < len; p++) { var lo = Math.Max(0, p - ctx); m.TrainStep(ids[lo..p], ids[p], lr); }   // teacher-forced windowed next-char
            }
            // teacher-forced accuracy = raw memorisation (given the TRUE window, does it predict the next char?)
            var tfOk = 0; for (var p = 1; p < len; p++) { var lo = Math.Max(0, p - ctx); if (m.Predict(ids[lo..p]) == ids[p]) tfOk++; }
            var tf = tfOk / (double)(len - 1);
            // free-run: prime, then feed own output through a sliding ctx window (robustness — one slip cascades)
            var gen = new List<int>(ids[..prime]);
            for (var p = prime; p < len; p++) { var lo = Math.Max(0, gen.Count - ctx); gen.Add(m.Predict(gen.GetRange(lo, gen.Count - lo).ToArray())); }
            var run = 0; while (prime + run < len && gen[prime + run] == ids[prime + run]) run++;
            var matched = 0; for (var p = prime; p < len; p++) if (gen[p] == ids[p]) matched++;
            res[si] = (S, m.ParamCount, tf, run, matched / (double)(len - prime));
        });

        foreach (var r in res) Console.WriteLine($"  {r.S,5}{r.p,12:N0}{r.tf,11:P0}{r.run,13}{r.match,9:P0}");
        Console.WriteLine($"\n  read: correct-run and match% RISING with S (especially once past ctx={ctx}) = rank helps it hold the thread beyond its window.");
        Console.WriteLine("  flat = rank isn't the lever for long-range coherence here; tracking is bounded by something else (depth or residual width).");
        Console.WriteLine($"  the full sequence is {len} chars, so a correct-run well above {ctx + prime} means it reproduced content it could no longer see.");
    }
}
