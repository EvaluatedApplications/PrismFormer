// Copyright (c) 2026 Dongyang Stephen Chen, trading as Evaluated Applications. All rights reserved.
// Source-available for study and verification; commercial use, redistribution, or derivatives require permission. See LICENSE.

using System.Collections.Concurrent;
using PrismFormer;

namespace PrismFormer.Bench;

/// <summary>
/// --crosstalk : the HRR/VSA CAPACITY WALL, measured, not assumed. A single dim-d phasor face is used as an associative
/// memory: <c>memory = Σ bind(key_i, value_i)</c> over K random pairs drawn from a V-face codebook. We retrieve each
/// value via <c>decode(unbind(memory, key_i))</c> (nearest codebook face by correlation) and find K_max = the most pairs
/// that still decode at ≥95%. That K_max is the DEPTH OF COMPOSITION one face supports at a given (dim, vocab): stack more
/// bound state than that into a face — whether across reasoning steps or a fuller bundle — and crosstalk breaks it.
///
/// Sweeps dim × vocab so you can read off, for the reset model, how deep you can compose before you must widen the face.
/// The fit check tests whether K_max ~ d / ln(V) (linear in dim, logarithmic in vocab) and prints the constant.
/// </summary>
public static class CrosstalkBench
{
    public static void Run()
    {
        try { Console.OutputEncoding = System.Text.Encoding.UTF8; } catch { }
        Console.WriteLine($"HRR CROSSTALK CAPACITY — bound pairs per face before decode breaks   {DateTime.Now:yyyy-MM-dd HH:mm}");
        Console.WriteLine("  memory = Σ bind(key_i, value_i)  ·  retrieve value_j = argmax-correlate(unbind(memory, key_j), codebook)");
        Console.WriteLine("  K_max = most pairs still decoding ≥95%  =  composition depth one dim-d face supports over a vocab-V codebook\n");

        var dims = new[] { 16, 32, 48, 64, 96, 128, 192, 256 };   // include the TINY end — does even 32/48/64 hold enough?
        var vocabs = new[] { 256, 1024, 4096 };
        const int trials = 30; const double thresh = 0.95;

        static double[] RandFace(Random r, int d) { var f = new double[d]; for (var i = 0; i < d; i += 2) { var a = r.NextDouble() * 2 * Math.PI; f[i] = Math.Cos(a); f[i + 1] = Math.Sin(a); } return f; }
        static double Dot(double[] a, double[] b) { double s = 0; for (var i = 0; i < a.Length; i++) s += a[i] * b[i]; return s; }
        static int Decode(double[] q, double[][] cb) { var best = 0; var bv = double.NegativeInfinity; for (var j = 0; j < cb.Length; j++) { var s = Dot(q, cb[j]); if (s > bv) { bv = s; best = j; } } return best; }

        var cells = (from d in dims from V in vocabs select (d, V)).ToArray();
        var kmax = new ConcurrentDictionary<(int d, int V), int>();

        Parallel.ForEach(cells, cell =>
        {
            var (d, V) = cell;
            var rng = new Random(12345 + d * 31 + V);
            var cb = Enumerable.Range(0, V).Select(_ => RandFace(rng, d)).ToArray();
            var best = 0;
            for (var K = 1; K <= Math.Min(V, 256); K++)
            {
                var ok = 0; var tot = 0;
                for (var t = 0; t < trials; t++)
                {
                    var tr = new Random(1000 + t);
                    var mem = new double[d];
                    var keys = new int[K]; var vals = new int[K];
                    for (var i = 0; i < K; i++)
                    {
                        keys[i] = tr.Next(V); vals[i] = tr.Next(V);
                        var b = PhasorCodec.Bind(cb[keys[i]], cb[vals[i]]);
                        for (var x = 0; x < d; x++) mem[x] += b[x];
                    }
                    for (var i = 0; i < K; i++) { if (Decode(PhasorCodec.Unbind(mem, cb[keys[i]]), cb) == vals[i]) ok++; tot++; }
                }
                if (ok / (double)tot >= thresh) best = K; else break;
            }
            kmax[cell] = best;
        });

        Console.Write($"  {"K_max",-8}"); foreach (var V in vocabs) Console.Write($"{("V=" + V),9}"); Console.WriteLine();
        foreach (var d in dims) { Console.Write($"  d={d,-5} "); foreach (var V in vocabs) Console.Write($"{kmax[(d, V)],9}"); Console.WriteLine(); }

        Console.WriteLine("\n  fit — K_max·ln(V)/d  (roughly constant ⇒ capacity ~ d/ln(V); the number is the constant):");
        foreach (var d in dims) { Console.Write($"  d={d,-5} "); foreach (var V in vocabs) Console.Write($"{kmax[(d, V)] * Math.Log(V) / d,9:F2}"); Console.WriteLine(); }

        Console.WriteLine("\n  read: K_max is how many bound items ONE face holds. In the model, a reasoning chain that accumulates bound");
        Console.WriteLine("  state in a working face breaks past ~K_max steps unless the state is externalised (scratchpad) or dim is widened.");
    }
}
