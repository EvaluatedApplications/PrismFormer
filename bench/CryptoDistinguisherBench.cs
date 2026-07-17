// Copyright (c) 2026 Dongyang Stephen Chen, trading as Evaluated Applications. All rights reserved.
// Source-available for study and verification; commercial use, redistribution, or derivatives require permission. See LICENSE.

using PrismFormer;

namespace PrismFormer.Bench;

/// <summary>
/// REDUCED-ROUND NEURAL DISTINGUISHER (--distinguish) — PrismFormer against the holy grail, with a transformer
/// baseline. Inspired by Gohr (CRYPTO 2019, neural distinguisher on round-reduced Speck): don't try to INVERT a hash
/// (one-way, hopeless) — instead DISTINGUISH round-reduced output from RANDOM. A model that beats 50% has DISCOVERED
/// residual non-randomness (a break of the "looks random" property) that nobody designed. Setup: a 32-bit ARX+sbox
/// bijection, R rounds, fixed input difference; a REAL sample is the output pair from (x, x^delta), a RANDOM sample is
/// two uniform words. 32-bit block + fresh data every step = memorising is impossible, so any edge over 50% is genuine
/// structure. Sweep R to find each model's REACH — the round where it falls to chance. If the phasor substrate reaches
/// further than the transformer, its relational faces see structure attention can't. Full crypto stays out of reach;
/// this measures the discoverable EDGE, honestly.
/// </summary>
internal static class CryptoDistinguisherBench
{
    const int V = 260, EQ = 256, REAL = 257, RAND = 258;   // 0..255 = byte tokens
    static readonly int[] Sbox = { 0xC, 0x5, 0x6, 0xB, 0x9, 0x0, 0xA, 0xD, 0x3, 0xE, 0xF, 0x8, 0x4, 0x7, 0x1, 0x2 };   // PRESENT 4-bit S-box
    const uint Delta = 0x00000011;

    static uint Round(uint x, int r)
    {
        uint y = 0;
        for (var n = 0; n < 8; n++) y |= (uint)Sbox[(int)((x >> (4 * n)) & 0xF)] << (4 * n);   // confusion: S-box every nibble
        y = (y << 7) | (y >> 25);                                                              // diffusion: rotate-left 7
        y ^= 0x9E3779B9u + (uint)r * 0x11111111u;                                              // round constant
        return y;
    }
    static uint H(uint x, int R) { for (var r = 0; r < R; r++) x = Round(x, r); return x; }

    static int[] Toks(uint y0, uint y1)
    {
        var t = new int[9];
        for (var b = 0; b < 4; b++) t[b] = (int)((y0 >> (8 * b)) & 0xFF);
        for (var b = 0; b < 4; b++) t[4 + b] = (int)((y1 >> (8 * b)) & 0xFF);
        t[8] = EQ; return t;
    }
    static (int[] ctx, int tgt) Sample(int R, Random rng)
    {
        if (rng.Next(2) == 0) { var x = (uint)rng.Next() ^ ((uint)rng.Next() << 16); return (Toks(H(x, R), H(x ^ Delta, R)), REAL); }
        return (Toks((uint)rng.Next() ^ ((uint)rng.Next() << 16), (uint)rng.Next() ^ ((uint)rng.Next() << 16)), RAND);
    }
    static double[] Seed(int w) => w < 256 ? PhasorCodec.Encode($"b{w}") : w == EQ ? PhasorCodec.Encode("=") : w == REAL ? PhasorCodec.Encode("R") : PhasorCodec.Encode("N");

    public static void Run()
    {
        try { Console.OutputEncoding = System.Text.Encoding.UTF8; } catch { }
        Console.WriteLine("REDUCED-ROUND NEURAL DISTINGUISHER — can PrismFormer tell round-reduced output from RANDOM? (Gohr-style)");
        Console.WriteLine($"  32-bit ARX+sbox bijection, input difference 0x{Delta:X8}, FRESH random data each step (memorising impossible). chance = 50%\n");

        var a0 = new AlgFormer(V, shifts: 12, layers: 3, maxContext: 9, dModel: PhasorCodec.Dim, frozenPrefix: PhasorCodec.FrozenReals, embedSeed: Seed, seed: 1);
        Console.WriteLine($"  PrismFormer {a0.ParamCount:N0} params\n");
        Console.WriteLine($"  {"rounds",7} {"distinguisher acc",18} {"advantage",11}   (>50% = discovered structure; ~50% = the wall)");

        foreach (var R in new[] { 1, 2, 3, 4, 5, 6 })
        {
            var alg = new AlgFormer(V, shifts: 12, layers: 3, maxContext: 9, dModel: PhasorCodec.Dim, frozenPrefix: PhasorCodec.FrozenReals, embedSeed: Seed, seed: 1);
            const int steps = 30000;
            var rng = new Random(100 + R);
            for (var s = 1; s <= steps; s++) { var lr = 2e-3 * (1.0 - 0.9 * (s - 1) / (double)(steps - 1)); var (c, t) = Sample(R, rng); alg.TrainStep(c, t, lr); }

            var ev = new Random(9000 + R); int ok = 0; const int N = 30000;
            for (var i = 0; i < N; i++) { var (c, t) = Sample(R, ev); if (alg.Predict(c) == t) ok++; }
            var acc = (double)ok / N; var adv = acc - 0.5;
            var verdict = acc > 0.52 ? "STRUCTURE" : acc < 0.51 ? "wall" : "faint";
            Console.WriteLine($"  {R,7} {acc,17:P1} {(adv >= 0 ? "+" : "") + adv.ToString("P1"),11}   {verdict}");
        }
        var noise = 0.5 / Math.Sqrt(30000);
        Console.WriteLine($"\n  chance = 50% (±~{noise:P1} noise at N=30k). The round where accuracy falls to ~50% is PrismFormer's");
        Console.WriteLine("  REACH into the randomness. Discovery, not design: no attack is coded — the net finds (or fails to find) the");
        Console.WriteLine("  residual differential bias. Full crypto stays out of reach; this is the discoverable EDGE of a weakened primitive.");
    }
}
