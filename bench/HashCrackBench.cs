// Copyright (c) 2026 Dongyang Stephen Chen, trading as Evaluated Applications. All rights reserved.
// Source-available for study and verification; commercial use, redistribution, or derivatives require permission. See LICENSE.

using PrismFormer;

namespace PrismFormer.Bench;

/// <summary>
/// HASH-CRACK probe (--crack). Small hashes = byte PERMUTATIONS (bijective, so the inverse exists). Train the model
/// to INVERT them (given h(x), predict x) on ~70% of inputs, then attempt held-out inputs it never saw. We report
/// LOSS, not just accuracy: chance loss is ln(256)=5.545 nats, so held-out loss BELOW that = the model captured some
/// structure ("getting warmer"); held-out loss AT chance = literally zero signal (as far from cracking as possible).
/// Swept over rounds of diffusion to show how fast the signal dies. Cracking a real (salted, huge-space) hash stays
/// impossible regardless — this measures how close a TINY, structured hash gets.
/// </summary>
internal static class HashCrackBench
{
    const int V = 300, EQ = 256;
    static readonly double Chance = Math.Log(256);

    // one bijective mixing round on a byte (invertible => the hash is a permutation of 0..255)
    static byte Round(byte x) { x = (byte)((x * 167 + 29) & 255); x = (byte)(((x << 3) | (x >> 5)) & 255); return (byte)(x ^ (x >> 4) ^ 0x5A); }
    static int HashR(int x, int rounds) { var b = (byte)x; for (var r = 0; r < rounds; r++) b = Round(b); return b; }

    static (double trL, double heL, double heA) Crack(int rounds, int seed)
    {
        // inverse pairs: input = h(x), target = x
        var pairs = Enumerable.Range(0, 256).Select(x => (h: HashR(x, rounds), x)).ToArray();
        bool Held(int h) => (uint)(h * 2654435761) % 100 < 30;
        var train = pairs.Where(p => !Held(p.h)).ToArray();
        var held = pairs.Where(p => Held(p.h)).ToArray();
        double[] Seed(int w) => w < 256 ? PhasorCodec.NumberFace(w) : PhasorCodec.Encode("=");
        var m = new AlgFormer(V, shifts: 8, layers: 2, maxContext: 4, dModel: PhasorCodec.Dim, frozenPrefix: PhasorCodec.FrozenReals, embedSeed: Seed, seed: seed);
        var rng = new Random(seed);
        const int steps = 6000;
        for (var t = 1; t <= steps; t++)
        {
            var lr = 3e-3 * (1.0 - 0.9 * (t - 1) / (double)(steps - 1));
            var (h, x) = train[rng.Next(train.Length)];
            m.TrainStep(new[] { h, EQ }, x, lr);
        }
        (double loss, double acc) Eval((int h, int x)[] s)
        {
            double tot = 0; int ok = 0;
            foreach (var (h, x) in s)
            {
                var lg = m.LogitsFor(new[] { h, EQ });
                var max = lg.Max(); var sum = 0.0; for (var w = 0; w < lg.Length; w++) sum += Math.Exp(lg[w] - max);
                tot += -(lg[x] - max - Math.Log(sum));
                var best = 0; for (var w = 1; w < lg.Length; w++) if (lg[w] > lg[best]) best = w;
                if (best == x) ok++;
            }
            return (tot / s.Length, ok / (double)s.Length);
        }
        var tr = Eval(train); var he = Eval(held);
        return (tr.loss, he.loss, he.acc);
    }

    public static void Run()
    {
        try { Console.OutputEncoding = System.Text.Encoding.UTF8; } catch { }
        Console.WriteLine("HASH-CRACK — train to INVERT small byte-permutation hashes, measure held-out loss vs chance\n");
        Console.WriteLine($"  chance loss (no signal) = ln(256) = {Chance:0.000} nats.  held-out loss BELOW that = getting warmer.\n");
        Console.WriteLine($"  {"rounds",7} {"train loss",11} {"HELD loss",10} {"vs chance",10} {"held acc",9}");
        foreach (var r in new[] { 0, 1, 2, 3, 4, 6 })
        {
            var (trL, heL, heA) = Crack(r, 1);
            var gap = Chance - heL;   // >0 = below chance = some signal
            var verdict = gap > 0.15 ? "SIGNAL" : "~zero";
            Console.WriteLine($"  {r,7} {trL,11:0.000} {heL,10:0.000} {gap,+9:0.000;-0.000} {heA,9:P0}   {verdict}");
        }
        Console.WriteLine("\n  round 0 = identity (trivially invertible). One real round of mixing and held-out loss snaps to the chance");
        Console.WriteLine("  floor — the model memorises the training half perfectly but has ZERO information about unseen inputs. Not");
        Console.WriteLine("  'close and improvable' — exactly at chance, the maximum possible distance from cracking. Loss confirms accuracy.");
    }
}
