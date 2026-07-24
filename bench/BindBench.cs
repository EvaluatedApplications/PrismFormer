// Copyright (c) 2026 Dongyang Stephen Chen, trading as Evaluated Applications. All rights reserved.
// Source-available for study and verification; commercial use, redistribution, or derivatives require permission. See LICENSE.

using PrismFormer;

namespace PrismFormer.Bench;

/// <summary>
/// --bind : does the FROZEN codec short-cut to a token that was NEVER bound? Where --crosstalk uses RANDOM faces and reports
/// only K_max, this uses the REAL frozen codebook — every vocab token's <see cref="PhasorCodec.Encode"/> face, exactly what
/// the model seeds — and at each DEPTH builds memory = Σ bind(key,value), unbinds every key, decodes the result to the
/// nearest real face, and splits the outcome three ways:
///   RECALL       — decoded the right value (clean),
///   OTHER-BOUND  — decoded a different token that WAS in the bundle (crosstalk within the set),
///   PHANTOM      — decoded a token that was NEVER bound at all (the "wrong codec" the theory predicts).
/// It also confirms a pure bind CHAIN is exact, so the fidelity limit is a BUNDLING (superposition) effect, not a binding
/// one. Recall sliding down smoothly + phantoms rising smoothly = the deterministic-crosstalk regime; a sudden cliff = a
/// hard fidelity wall at that depth. Real codec vs random (--crosstalk) tells you whether the codec STRUCTURE helps or hurts.
/// </summary>
public static class BindBench
{
    public static void Run(int maxDepth = 14, int trials = 48)
    {
        try { Console.OutputEncoding = System.Text.Encoding.UTF8; } catch { }
        var vocab = new CharVocab();
        int V = vocab.Size;
        Console.WriteLine($"FROZEN codec crosstalk — real codebook of {V:N0} token faces · d{PhasorCodec.Dim} · {DateTime.Now:yyyy-MM-dd HH:mm}\n");
        var faces = new double[V][];
        for (var id = 0; id < V; id++) faces[id] = PhasorCodec.Encode(vocab.Symbol(id));

        int Decode(double[] h) { var best = 0; var bm = double.NegativeInfinity; for (var i = 0; i < V; i++) { var c = PhasorCodec.Correlate(h, faces[i]); if (c > bm) { bm = c; best = i; } } return best; }

        // 1) a pure BIND chain is lossless (complex multiply is exactly invertible per component) — crosstalk is a BUNDLING effect
        var r0 = new Random(7);
        var chain = Enumerable.Range(0, 8).Select(_ => r0.Next(V)).ToArray();
        var bound = faces[chain[0]];
        for (var i = 1; i < chain.Length; i++) bound = PhasorCodec.Bind(bound, faces[chain[i]]);
        var back = bound;
        for (var i = 1; i < chain.Length; i++) back = PhasorCodec.Unbind(back, faces[chain[i]]);   // unbind the 7 others → should leave the first
        Console.WriteLine($"pure bind chain of 8 faces → unbind 7, decode the remainder: {(Decode(back) == chain[0] ? "EXACT ✓" : "BROKE ✗")}  (binding alone is lossless — crosstalk is a BUNDLING effect)\n");

        // 2) key→value store: bundle D binds, unbind each key, decode the value against the WHOLE real codebook
        Console.WriteLine("KEY→VALUE store: memory = Σ bind(key,value) over D pairs; unbind each key; decode nearest of all " + $"{V:N0} faces");
        Console.WriteLine($"  {"depth",6}{"recall",10}{"other-bound",14}{"PHANTOM",10}   (phantom = a token that was NEVER bound)");
        for (var depth = 1; depth <= maxDepth; depth++)
        {
            long correct = 0, other = 0, phantom = 0, total = 0; var gate = new object();
            Parallel.For(0, trials, t =>
            {
                var lr = new Random(1000 + t + depth * 7919);
                var pick = DistinctK(lr, V, 2 * depth);
                var keys = pick[..depth]; var vals = pick[depth..];
                var boundSet = new HashSet<int>(pick);
                var binds = new double[depth][];
                for (var i = 0; i < depth; i++) binds[i] = PhasorCodec.Bind(faces[keys[i]], faces[vals[i]]);
                var mem = PhasorCodec.Bundle(binds);
                long c = 0, o = 0, p = 0;
                for (var i = 0; i < depth; i++)
                {
                    var got = Decode(PhasorCodec.Unbind(mem, faces[keys[i]]));
                    if (got == vals[i]) c++;
                    else if (boundSet.Contains(got)) o++;   // a different token that WAS in the bundle
                    else p++;                                // a token that was never bound
                }
                lock (gate) { correct += c; other += o; phantom += p; total += depth; }
            });
            Console.WriteLine($"  {depth,6}{correct / (double)total,10:P0}{other / (double)total,14:P0}{phantom / (double)total,10:P0}");
        }
        Console.WriteLine("\nread: recall sliding down + PHANTOM rising SMOOTHLY = deterministic crosstalk the attenuation could learn to use.");
        Console.WriteLine("a sudden cliff = a hard fidelity wall at that depth. Compare with --crosstalk (random faces) to see if the codec's STRUCTURE helps or hurts.");
    }

    static int[] DistinctK(Random r, int n, int k) { var s = new HashSet<int>(); while (s.Count < k) s.Add(r.Next(n)); return s.ToArray(); }
}
