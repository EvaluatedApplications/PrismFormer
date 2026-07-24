// Copyright (c) 2026 Dongyang Stephen Chen, trading as Evaluated Applications. All rights reserved.
// Source-available for study and verification; commercial use, redistribution, or derivatives require permission. See LICENSE.

using PrismFormer;

namespace PrismFormer.Bench;

/// <summary>
/// --mixedspec : CAN PETS OF DIFFERENT SIZE STILL SHARE? Two models grown from ONE common ancestor (so their overlapping
/// layers are aligned), then specialised: A is grown DEEP (L6) and trained on skill X, B stays SHALLOW (L3) and trained on
/// skill Y, same codec. We elastic-average only the OVERLAP (first min(L) layers, shared shifts, shared positions) via
/// AlgFormer.PartialAverage and check: does each keep its OWN skill (the merge didn't break it) and pick up the OTHER's (skill
/// transported across the depth mismatch through the shared layers)? If yes, mismatched-spec pets can average/breed without a
/// hard-fork gate. 'common' is what the shared ancestor knew — it should survive on both.
/// </summary>
public static class MixedSpecBench
{
    public static void Run(int passes = 100)
    {
        try { Console.OutputEncoding = System.Text.Encoding.UTF8; } catch { }
        int d = PhasorCodec.Dim;
        const int V = 64, ctx = 8, nFacts = 40;
        double[] Seed(int w) { var f = PhasorCodec.Encode("m" + w); if (f.Length == d) return f; var s = new double[d]; Array.Copy(f, s, Math.Min(f.Length, d)); return s; }

        var rng = new Random(3);
        (int[] c, int t)[] Make(int n) { var f = new (int[], int)[n]; for (var i = 0; i < n; i++) { var c = new int[4]; for (var j = 0; j < 4; j++) c[j] = rng.Next(V); f[i] = (c, rng.Next(V)); } return f; }
        var common = Make(nFacts); var X = Make(nFacts); var Y = Make(nFacts);
        void Train(AlgFormer m, (int[] c, int t)[] f, int p) { for (var ep = 0; ep < p; ep++) { var lr = 3e-3 * (1.0 - 0.8 * ep / Math.Max(1, p - 1)); foreach (var (c, t) in f.OrderBy(_ => rng.Next())) m.TrainStep(c, t, lr); } }
        double Rec(AlgFormer m, (int[] c, int t)[] f) { var ok = 0; foreach (var (c, t) in f) if (m.Predict(c) == t) ok++; return ok / (double)f.Length; }

        var baseM = new AlgFormer(V, shifts: 8, layers: 3, maxContext: ctx, dModel: d, frozenPrefix: d, embedSeed: Seed, seed: 1);
        Train(baseM, common, passes);                                   // shared ancestor
        var A = baseM.GrowLayers(3, zeroOutputOnly: true, seed: 2);      // A: grown DEEP (L6), first 3 layers = the ancestor
        var B = AlgFormer.Deserialize(baseM.Serialize());               // B: stays SHALLOW (L3), a clone of the ancestor
        Train(A, X, passes);                                            // A learns skill X (all 6 layers adapt)
        Train(B, Y, passes);                                            // B learns skill Y (3 layers adapt)

        Console.WriteLine($"MIXED-SPEC AVERAGING — deep A (L{A.Layers}) + shallow B (L{B.Layers}), shared codec, averaged on the overlap\n");
        Console.WriteLine($"  {"model",-22}{"common",9}{"own",8}{"other",9}");
        Console.WriteLine($"  {"A (L6)  before",-22}{Rec(A, common),9:P0}{Rec(A, X),8:P0}{Rec(A, Y),9:P0}");
        Console.WriteLine($"  {"B (L3)  before",-22}{Rec(B, common),9:P0}{Rec(B, Y),8:P0}{Rec(B, X),9:P0}");

        var Asnap = AlgFormer.Deserialize(A.Serialize());               // symmetric merge: B averages against pre-merge A
        A.PartialAverage(B, 0.5);
        B.PartialAverage(Asnap, 0.5);

        Console.WriteLine($"  {"A (L6)  after",-22}{Rec(A, common),9:P0}{Rec(A, X),8:P0}{Rec(A, Y),9:P0}");
        Console.WriteLine($"  {"B (L3)  after",-22}{Rec(B, common),9:P0}{Rec(B, Y),8:P0}{Rec(B, X),9:P0}");

        Console.WriteLine("\n  read: 'own' staying up after the merge = averaging across the depth mismatch didn't break each pet;");
        Console.WriteLine("  'other' rising = the skill TRANSPORTED through the shared layers. Both = mixed-spec pets can share without a fork.");
        Console.WriteLine("  (alpha 0.5 is one aggressive shot; the live swarm uses 0.05, far gentler — this is the stress test.)");
    }
}
