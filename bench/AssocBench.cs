// Copyright (c) 2026 Dongyang Stephen Chen, trading as Evaluated Applications. All rights reserved.
// Source-available for study and verification; commercial use, redistribution, or derivatives require permission. See LICENSE.

using PrismFormer;

namespace PrismFormer.Bench;

/// <summary>
/// DISORDER-CODEC associative memory (--assoc). Tests the idea: use a STRUCTURELESS codec (each token -> an
/// orthogonal/hashed face) and store hash pairs as a VSA associative memory  M = bundle( bind(key(x), val(h(x))) ).
/// Then REVERSE-recall: given h(x), unbind M by val(h) -> ~key(x) -> clean up to x. This inverts a hash — but only
/// for STORED pairs (it's a precomputed table / rainbow table), and capacity is bounded (crosstalk grows with the
/// number of pairs). No training. Shows exactly what "match the hash's disorder" buys: perfect MEMORY, not cracking
/// — memory recalls what you already computed forward; it can't answer for inputs you never stored (0%).
/// </summary>
internal static class AssocBench
{
    static byte Round(byte x) { x = (byte)((x * 167 + 29) & 255); x = (byte)(((x << 3) | (x >> 5)) & 255); return (byte)(x ^ (x >> 4) ^ 0x5A); }
    static int Hash(int x) { var b = (byte)(x & 255); for (var r = 0; r < 6; r++) b = Round(b); return b | ((x >> 8) << 8); }   // 6-round mix on low byte, keep high bits distinct

    // structureless codec: a distinct orthogonal-ish phasor face per id (hashed signature + orbital)
    static double[] Key(int x) => PhasorCodec.Encode($"k{x}");
    static double[] Val(int h) => PhasorCodec.Encode($"v{h}");

    public static void Run()
    {
        try { Console.OutputEncoding = System.Text.Encoding.UTF8; } catch { }
        Console.WriteLine("DISORDER-CODEC associative memory — reverse hash-lookup by VSA cleanup (no training)\n");
        Console.WriteLine("store M = bundle( bind(key(x), val(h(x))) );  reverse: unbind M by val(h) -> cleanup -> x\n");
        Console.WriteLine($"  {"stored pairs",13} {"reverse-recall (stored)",24} {"unseen h",10}");
        var rng = new Random(3);
        foreach (var K in new[] { 5, 20, 50, 100, 200, 400 })
        {
            // K distinct inputs and their hashes
            var xs = new List<int>(); var seen = new HashSet<int>();
            while (xs.Count < K) { var x = rng.Next(100000); if (seen.Add(x)) xs.Add(x); }
            var hs = xs.Select(Hash).ToArray();

            // build the associative memory
            var M = new double[PhasorCodec.Dim];
            for (var i = 0; i < K; i++) { var b = PhasorCodec.Bind(Key(xs[i]), Val(hs[i])); for (var d = 0; d < M.Length; d++) M[d] += b[d]; }

            // reverse recall: given each stored hash, recover its input by cleanup over the stored keys
            int ok = 0;
            for (var i = 0; i < K; i++)
            {
                var cand = PhasorCodec.Unbind(M, Val(hs[i]));           // ~ Key(xs[i]) + crosstalk
                var best = 0; var bz = double.NegativeInfinity;
                for (var j = 0; j < K; j++) { var s = PhasorCodec.Correlate(cand, Key(xs[j])); if (s > bz) { bz = s; best = j; } }
                if (xs[best] == xs[i]) ok++;
            }

            // unseen: hash of an input NOT stored -> can it recover anything valid? (chance / 0)
            int okU = 0; const int trials = 200;
            for (var t = 0; t < trials; t++)
            {
                int x; do { x = rng.Next(100000); } while (seen.Contains(x));
                var cand = PhasorCodec.Unbind(M, Val(Hash(x)));
                var best = 0; var bz = double.NegativeInfinity;
                for (var j = 0; j < K; j++) { var s = PhasorCodec.Correlate(cand, Key(xs[j])); if (s > bz) { bz = s; best = j; } }
                // "recovered" x would have to equal the true x — impossible, it isn't even a candidate. count as fail always.
                if (xs[best] == x) okU++;   // can never happen: x not in the stored set
            }
            Console.WriteLine($"  {K,13} {(double)ok / K,22:P0}   {(double)okU / trials,9:P1}");
        }
        Console.WriteLine("\n  your idea WORKS: a disorder codec reverse-inverts the hashes it STORED (capacity-limited by crosstalk).");
        Console.WriteLine("  but that's a rainbow table — it only inverts what was computed forward and stored. Unseen inputs: 0%, always");
        Console.WriteLine("  (they aren't even candidates). Perfect MEMORY, not cracking. Salt or a large input space defeats it entirely.");
    }
}
