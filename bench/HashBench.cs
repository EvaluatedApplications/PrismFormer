// Copyright (c) 2026 Dongyang Stephen Chen, trading as Evaluated Applications. All rights reserved.
// Source-available for study and verification; commercial use, redistribution, or derivatives require permission. See LICENSE.

using PrismFormer;

namespace PrismFormer.Bench;

/// <summary>
/// HASH LEARNABILITY probe (--hash). Can the model LEARN a byte-mixing function (generalise to unseen inputs),
/// or only MEMORISE the table? Train on ~70% of the 256 inputs, test HELD-OUT on the rest. A structured map
/// (modular add) should generalise via the codec's modular-add homomorphism; as we add rounds of diffusion
/// (XOR/rotate/multiply), the map avalanches and held-out collapses to chance — the model memorises but cannot
/// learn. Empirically maps where structure dies. (Forward learnability only; a one-way hash still can't be
/// inverted regardless — this is why "crack a hash" is impossible, not merely hard.)
/// </summary>
internal static class HashBench
{
    const int V = 300, EQ = 256;

    static byte Round(byte x)
    {
        x = (byte)((x * 167 + 29) & 255);            // affine mix
        x = (byte)(((x << 3) | (x >> 5)) & 255);     // rotate left 3
        x = (byte)(x ^ (x >> 4) ^ 0x5A);             // xor-shift + constant
        return x;
    }
    static int HashR(int x, int rounds) { var b = (byte)x; for (var r = 0; r < rounds; r++) b = Round(b); return b; }
    static int Add(int x) => (x + 37) & 255;

    static (double tr, double he) Learn(Func<int, int> f, int seed)
    {
        var pairs = Enumerable.Range(0, 256).Select(x => (x, y: f(x))).ToArray();
        bool Held(int x) => (uint)(x * 2654435761) % 100 < 30;   // stable ~30% held-out by input
        var train = pairs.Where(p => !Held(p.x)).ToArray();
        var held = pairs.Where(p => Held(p.x)).ToArray();
        double[] Seed(int w) => w < 256 ? PhasorCodec.NumberFace(w) : PhasorCodec.Encode("=");
        var m = new AlgFormer(V, shifts: 8, layers: 2, maxContext: 4, dModel: PhasorCodec.Dim, frozenPrefix: PhasorCodec.FrozenReals, embedSeed: Seed, seed: seed);
        var rng = new Random(seed);
        const int steps = 5000;
        for (var t = 1; t <= steps; t++)
        {
            var lr = 3e-3 * (1.0 - 0.9 * (t - 1) / (double)(steps - 1));
            var (x, y) = train[rng.Next(train.Length)];
            m.TrainStep(new[] { x, EQ }, y, lr);
        }
        double Acc((int x, int y)[] s) { var ok = 0; foreach (var (x, y) in s) if (m.Predict(new[] { x, EQ }) == y) ok++; return ok / (double)s.Length; }
        return (Acc(train), Acc(held));
    }

    public static void Run()
    {
        try { Console.OutputEncoding = System.Text.Encoding.UTF8; } catch { }
        Console.WriteLine("HASH LEARNABILITY — held-out generalisation vs diffusion (learn the map, or only memorise?)\n");
        Console.WriteLine($"  {"function",-22} {"train",7} {"HELD-OUT",9}   (held-out chance ~ 0.4%)");
        var tasks = new (string name, Func<int, int> f)[]
        {
            ("add (x+37) mod 256",  Add),
            ("mix, 1 round",        x => HashR(x, 1)),
            ("mix, 2 rounds",       x => HashR(x, 2)),
            ("mix, 3 rounds",       x => HashR(x, 3)),
            ("mix, 6 rounds (hash)",x => HashR(x, 6)),
        };
        foreach (var (name, f) in tasks)
        {
            var (tr, he) = Learn(f, 1);
            Console.WriteLine($"  {name,-22} {tr,7:P0} {he,9:P1}");
        }
        Console.WriteLine("\n  add generalises (codec's modular-add homomorphism); more diffusion -> held-out collapses to chance.");
        Console.WriteLine("  the model MEMORISES the table but cannot LEARN the mixing. No codec fixes it: a hash has no structure to");
        Console.WriteLine("  encode, and a homomorphic codec would only compute it FORWARD — cracking needs to invert a one-way function.");
    }
}
