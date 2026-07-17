// Copyright (c) 2026 Dongyang Stephen Chen, trading as Evaluated Applications. All rights reserved.
// Source-available for study and verification; commercial use, redistribution, or derivatives require permission. See LICENSE.

using PrismFormer;

namespace PrismFormer.Bench;

/// <summary>
/// SPECK32/64 NEURAL DISTINGUISHER (--speck) — PrismFormer on Gohr's actual benchmark (CRYPTO 2019). Real Speck32/64
/// (verified against the official test vector), input difference 0x0040/0x0000, REAL = ciphertext pair from a
/// random key + a plaintext pair with that difference, RANDOM = four uniform words; fresh data every step. Report
/// distinguisher accuracy at 5–8 rounds next to Gohr's published numbers (R5 92.9 / R6 78.8 / R7 61.6 / R8 51.4).
/// Honest: Gohr is a heavily-optimised deep ResNet on ~10^7 examples; a small relational model beating him at 7–8
/// rounds is a long shot — but the number on the REAL cipher is the point. Full/deployed crypto untouched.
/// </summary>
internal static class SpeckDistinguisherBench
{
    const int V = 32, EQ = 16, REAL = 17, RAND = 18;
    const ushort DX = 0x0040, DY = 0x0000;   // Gohr's input difference

    static ushort ROR(ushort x, int r) => (ushort)(((x >> r) | (x << (16 - r))) & 0xFFFF);
    static ushort ROL(ushort x, int r) => (ushort)(((x << r) | (x >> (16 - r))) & 0xFFFF);
    static (ushort x, ushort y) Rnd(ushort x, ushort y, ushort k)   // Speck32/64: alpha=7, beta=2
    {
        x = (ushort)((((ROR(x, 7) + y) & 0xFFFF) ^ k) & 0xFFFF);
        y = (ushort)(ROL(y, 2) ^ x);
        return (x, y);
    }
    static ushort[] KeySchedule(ushort[] key, int R)   // key = {k0, l0, l1, l2}
    {
        var k = new ushort[R]; var l = new ushort[R + 3];
        k[0] = key[0]; l[0] = key[1]; l[1] = key[2]; l[2] = key[3];
        for (var i = 0; i < R - 1; i++)
        {
            l[i + 3] = (ushort)((((ROR(l[i], 7) + k[i]) & 0xFFFF) ^ (ushort)i) & 0xFFFF);
            k[i + 1] = (ushort)(ROL(k[i], 2) ^ l[i + 3]);
        }
        return k;
    }
    static (ushort x, ushort y) Enc(ushort x, ushort y, ushort[] rk, int R) { for (var i = 0; i < R; i++) (x, y) = Rnd(x, y, rk[i]); return (x, y); }

    static bool VerifyTestVector()   // official Speck32/64: key 1918 1110 0908 0100, pt 6574 694c -> ct a868 42f2
    {
        var rk = KeySchedule(new ushort[] { 0x0100, 0x0908, 0x1110, 0x1918 }, 22);
        var (cx, cy) = Enc(0x6574, 0x694c, rk, 22);
        return cx == 0xa868 && cy == 0x42f2;
    }

    static ushort U16(Random r) => (ushort)r.Next(65536);
    static int[] Toks(ushort a, ushort b, ushort c, ushort d)   // 4 words -> 16 nibble tokens + EQ
    {
        var t = new int[17]; var p = 0;
        foreach (var w in new[] { a, b, c, d }) for (var n = 0; n < 4; n++) t[p++] = (w >> (4 * n)) & 0xF;
        t[16] = EQ; return t;
    }
    static (int[] ctx, int tgt) Sample(int R, Random rng)
    {
        if (rng.Next(2) == 0)
        {
            var rk = KeySchedule(new[] { U16(rng), U16(rng), U16(rng), U16(rng) }, R);
            ushort px = U16(rng), py = U16(rng);
            var (c0x, c0y) = Enc(px, py, rk, R);
            var (c1x, c1y) = Enc((ushort)(px ^ DX), (ushort)(py ^ DY), rk, R);
            return (Toks(c0x, c0y, c1x, c1y), REAL);
        }
        return (Toks(U16(rng), U16(rng), U16(rng), U16(rng)), RAND);
    }
    static double[] Seed(int w) => w < 16 ? PhasorCodec.Encode($"n{w}") : w == EQ ? PhasorCodec.Encode("=") : w == REAL ? PhasorCodec.Encode("R") : PhasorCodec.Encode("N");

    const int NTrain = 2_000_000, NTest = 40_000, Epochs = 5, Batch = 256;   // 10M views/round ~ Gohr's 10^7
    static int Shifts => 16;
    static int Layers => 5;
    static AlgFormer Model() => new(V, shifts: Shifts, layers: Layers, maxContext: 17, dModel: PhasorCodec.Dim, frozenPrefix: PhasorCodec.FrozenReals, embedSeed: Seed, seed: 1);
    static double Acc(AlgFormer m, (int[] c, int t)[] set) { var ok = 0; foreach (var (c, t) in set) if (m.Predict(c) == t) ok++; return ok / (double)set.Length; }

    public static void Run()
    {
        try { Console.OutputEncoding = System.Text.Encoding.UTF8; } catch { }
        Console.WriteLine("SPECK32/64 NEURAL DISTINGUISHER — Gohr's benchmark (CRYPTO 2019). input diff 0x0040/0x0000.");
        if (!VerifyTestVector()) { Console.WriteLine("  SPECK TEST VECTOR FAILED — cipher wrong, aborting (result would be meaningless)."); return; }
        Console.WriteLine("  Speck test vector PASSES (cipher correct).");
        Console.WriteLine($"  Gohr reference acc:  R5 92.9%   R6 78.8%   R7 61.6%   R8 51.4%   (his: deep ResNet, ~10^7 examples, GPU)");
        Console.WriteLine($"  PrismFormer {Model().ParamCount:N0} params | data-PARALLEL TrainEpoch on {Environment.ProcessorCount} cores | {NTrain:N0} samples x {Epochs} epochs = {(long)NTrain * Epochs:N0} views/round\n");
        Console.WriteLine($"  {"rounds",7} {"PrismFormer",13} {"Gohr",8} {"vs Gohr",9}");

        var gohr = new Dictionary<int, double> { { 5, 0.929 }, { 6, 0.788 }, { 7, 0.616 }, { 8, 0.514 } };
        foreach (var R in new[] { 5, 6, 7, 8 })
        {
            var gen = new Random(100 + R);
            var train = new (int[] Ctx, int Target)[NTrain];
            for (var i = 0; i < NTrain; i++) { var (c, t) = Sample(R, gen); train[i] = (c, t); }
            var evg = new Random(9000 + R);
            var test = new (int[] c, int t)[NTest];
            for (var i = 0; i < NTest; i++) { var (c, t) = Sample(R, evg); test[i] = (c, t); }

            var alg = Model();
            for (var ep = 1; ep <= Epochs; ep++)
            {
                var lr = 2e-3 * (1.0 - 0.85 * (ep - 1) / Math.Max(1, Epochs - 1));
                alg.TrainEpoch(train, Batch, lr, shuffleSeed: R * 1000 + ep);
                Console.WriteLine($"      R{R} epoch {ep}/{Epochs}: test {Acc(alg, test):P1}");
            }
            var acc = Acc(alg, test); var g = gohr[R]; var d = acc - g;
            Console.WriteLine($"  {R,7} {acc,12:P1} {g,7:P1} {(d >= 0 ? "+" : "") + d.ToString("P1"),9}   {(d >= 0 ? "*** BEATS GOHR ***" : "")}\n");
        }
        Console.WriteLine("  Beating Gohr = higher acc at 6-8 rounds. Full/deployed crypto untouched — reduced-round margin research,");
        Console.WriteLine("  and a distinguisher, not key recovery.");
    }
}
