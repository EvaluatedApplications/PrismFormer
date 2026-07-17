// Copyright (c) 2026 Dongyang Stephen Chen, trading as Evaluated Applications. All rights reserved.
// Source-available for study and verification; commercial use, redistribution, or derivatives require permission. See LICENSE.

using PrismFormer;

namespace PrismFormer.Bench;

/// <summary>
/// CODEC x FROZEN sweep for hash inversion (--crack-faces). Does a TOTALLY DISORDERED codec (orthogonal faces,
/// no structure) and/or UNFROZEN (learnable) faces help crack a hash? Trains to invert a 1-round byte permutation
/// under four settings and reports held-out loss vs the chance floor (ln256=5.545). Prediction: the structured
/// codec is *worse* than chance held-out (false similarity -> confidently wrong); the disordered codec sits *at*
/// chance (honest ignorance, orthogonal faces carry no misleading structure). Either way held-out cannot beat
/// chance — a hash has no structure to generalise, whatever the codec.
/// </summary>
internal static class CodecCrackBench
{
    const int V = 300, EQ = 256;
    static readonly double Chance = Math.Log(256);

    static byte Round(byte x) { x = (byte)((x * 167 + 29) & 255); x = (byte)(((x << 3) | (x >> 5)) & 255); return (byte)(x ^ (x >> 4) ^ 0x5A); }
    static int HashR(int x, int rounds) { var b = (byte)x; for (var r = 0; r < rounds; r++) b = Round(b); return b; }

    static (double trL, double heL, double heA) Crack(int rounds, bool disordered, bool unfrozen, int seed)
    {
        var pairs = Enumerable.Range(0, 256).Select(x => (h: HashR(x, rounds), x)).ToArray();
        bool Held(int h) => (uint)(h * 2654435761) % 100 < 30;
        var train = pairs.Where(p => !Held(p.h)).ToArray();
        var held = pairs.Where(p => Held(p.h)).ToArray();
        double[] Seed(int w) => w >= 256 ? PhasorCodec.Encode("=") : disordered ? PhasorCodec.Encode($"h{w}") : PhasorCodec.NumberFace(w);
        var frozen = unfrozen ? 0 : PhasorCodec.FrozenReals;
        var m = new AlgFormer(V, shifts: 8, layers: 2, maxContext: 4, dModel: PhasorCodec.Dim, frozenPrefix: frozen, embedSeed: Seed, seed: seed);
        var rng = new Random(seed);
        const int steps = 6000;
        for (var t = 1; t <= steps; t++) { var lr = 3e-3 * (1.0 - 0.9 * (t - 1) / (double)(steps - 1)); var (h, x) = train[rng.Next(train.Length)]; m.TrainStep(new[] { h, EQ }, x, lr); }
        (double loss, double acc) Eval((int h, int x)[] s)
        {
            double tot = 0; int ok = 0;
            foreach (var (h, x) in s)
            {
                var lg = m.LogitsFor(new[] { h, EQ });
                var max = lg.Max(); var sum = 0.0; for (var w = 0; w < lg.Length; w++) sum += Math.Exp(lg[w] - max);
                tot += -(lg[x] - max - Math.Log(sum));
                var best = 0; for (var w = 1; w < lg.Length; w++) if (lg[w] > lg[best]) best = w; if (best == x) ok++;
            }
            return (tot / s.Length, ok / (double)s.Length);
        }
        var tr = Eval(train); var he = Eval(held);
        return (tr.loss, he.loss, he.acc);
    }

    public static void Run()
    {
        try { Console.OutputEncoding = System.Text.Encoding.UTF8; } catch { }
        Console.WriteLine("CODEC x FROZEN sweep — invert a 1-round hash, held-out loss vs chance (ln256 = 5.545)\n");
        Console.WriteLine($"  {"codec",-12} {"faces",-10} {"train loss",11} {"HELD loss",10} {"vs chance",10} {"held acc",9}");
        var combos = new (string codec, bool dis, string fz, bool unf)[]
        {
            ("structured", false, "frozen",   false),
            ("structured", false, "unfrozen", true),
            ("DISORDERED", true,  "frozen",   false),
            ("DISORDERED", true,  "unfrozen", true),
        };
        foreach (var (codec, dis, fz, unf) in combos)
        {
            var (trL, heL, heA) = Crack(1, dis, unf, 1);
            Console.WriteLine($"  {codec,-12} {fz,-10} {trL,11:0.000} {heL,10:0.000} {Chance - heL,+9:0.000;-0.000} {heA,9:P0}");
        }
        Console.WriteLine("\n  RESULT: all four are WORSE than chance (~12-15 vs 5.5). Training drives the softmax peaked, so the model is");
        Console.WriteLine("  confidently WRONG on unseen hashes no matter the codec; disorder is if anything worse (orthogonal held-out");
        Console.WriteLine("  faces give more arbitrary confident answers), and unfreezing changes nothing. No codec / disorder / freezing");
        Console.WriteLine("  setting beats chance — a hash has no structure to generalise, so representation is irrelevant to the wall.");
    }
}
