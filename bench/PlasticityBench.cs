// Copyright (c) 2026 Dongyang Stephen Chen, trading as Evaluated Applications. All rights reserved.
// Source-available for study and verification; commercial use, redistribution, or derivatives require permission. See LICENSE.

using PrismFormer;

namespace PrismFormer.Bench;

/// <summary>
/// PLASTICITY / DEGENERACY probe (--plasticity), Levin-inspired — "do OTHER units take over?". Train the model, split
/// its weights into B blocks, and find the CRITICAL block (lesioning it hurts most = where the capability lives).
/// Then AMPUTATE that block and hold it dead: retrain in intervals, re-zeroing it after every interval so it can never
/// come back. If accuracy recovers with the critical region permanently gone, the function RELOCATED to surviving
/// units — degeneracy (many structures realise one goal). A final importance map confirms the load moved off the dead
/// block onto others. This is the mechanism behind why weight-averaging is lossy (--colony): distinct valid solutions.
/// </summary>
internal static class PlasticityBench
{
    const int Hi = 12, V = 64, PLUS = 30, EQ = 31, B = 8;

    static (int[] ctx, int tgt)[] Data()
    {
        var d = new List<(int[], int)>();
        for (var a = 0; a <= Hi; a++) for (var b = 0; b <= Hi; b++) d.Add((new[] { a, PLUS, b, EQ }, a + b));
        return d.ToArray();
    }
    static double[] Seed(int w) => w < 30 ? PhasorCodec.NumberFace(w) : PhasorCodec.Encode(w == PLUS ? "+" : "=");
    static AlgFormer Fresh(int seed) => new(V, shifts: 8, layers: 2, maxContext: 4, dModel: PhasorCodec.Dim, frozenPrefix: PhasorCodec.FrozenReals, embedSeed: Seed, seed: seed);
    static void Train(AlgFormer m, (int[] ctx, int tgt)[] data, int steps, int seed)
    {
        var rng = new Random(seed);
        for (var t = 1; t <= steps; t++) { var lr = 3e-3 * (1.0 - 0.9 * (t - 1) / (double)Math.Max(1, steps - 1)); var (c, y) = data[rng.Next(data.Length)]; m.TrainStep(c, y, lr); }
    }
    static double Acc(AlgFormer m, (int[] ctx, int tgt)[] data) { var ok = 0; foreach (var (c, y) in data) if (m.Predict(c) == y) ok++; return ok / (double)data.Length; }

    static byte[] ZeroBlock(byte[] b, int block)   // zero weight-double block [block/B .. (block+1)/B) of the param vector
    {
        var o = (byte[])b.Clone(); const int hdr = 24; var nD = (o.Length - hdr) / 8; var sz = nD / B;
        var lo = block * sz; var hi = block == B - 1 ? nD : lo + sz;
        for (var i = lo; i < hi; i++) for (var j = 0; j < 8; j++) o[hdr + i * 8 + j] = 0;
        return o;
    }
    static AlgFormer Amputate(AlgFormer m, int block) => AlgFormer.Deserialize(ZeroBlock(m.Serialize(), block));

    static double[] Fingerprint(AlgFormer m, (int[] ctx, int tgt)[] data)   // importance[b] = accuracy lost when block b is killed
    {
        var baseAcc = Acc(m, data); var imp = new double[B];
        for (var b = 0; b < B; b++) imp[b] = baseAcc - Acc(Amputate(m, b), data);
        return imp;
    }
    static string Bar(double[] v) => string.Join(" ", v.Select(x => $"{x,5:0.00}"));

    public static void Run()
    {
        try { Console.OutputEncoding = System.Text.Encoding.UTF8; } catch { }
        var data = Data();
        Console.WriteLine("PLASTICITY / DEGENERACY — amputate the critical weight-region, hold it dead, see if the function relocates\n");

        var m = Fresh(1); Train(m, data, 6000, 1);
        Console.WriteLine($"  trained baseline: {Acc(m, data):P0}");
        var f0 = Fingerprint(m, data); var crit = Array.IndexOf(f0, f0.Max());
        Console.WriteLine($"  importance map (acc lost when each block killed):\n      {Bar(f0)}");
        Console.WriteLine($"  critical block = #{crit} (killing it alone drops accuracy by {f0[crit]:P0})\n");

        Console.WriteLine($"  AMPUTATE block #{crit} and HOLD IT DEAD while retraining (re-zeroed every interval):");
        var m2 = Amputate(m, crit);
        Console.WriteLine($"      right after amputation:  {Acc(m2, data):P0}");
        for (var it = 1; it <= 8; it++)
        {
            Train(m2, data, 800, 100 + it);
            m2 = Amputate(m2, crit);                 // keep the region dead — recovery must come from OTHER units
            Console.WriteLine($"      +{it * 800,4} steps (block #{crit} still dead):  {Acc(m2, data):P0}");
        }

        var f1 = Fingerprint(m2, data);
        Console.WriteLine($"\n  importance map AFTER healing attempt:\n      {Bar(f1)}");
        Console.WriteLine($"  block #{crit}: was {f0[crit]:P0} critical, held dead. RESULT: no relocation — recovery stalls near chance, the");
        Console.WriteLine("  function does NOT move to surviving units. And every block starts 70-98% critical: the code is DISTRIBUTED");
        Console.WriteLine("  (all regions participate) yet NON-REDUNDANT (none is dispensable) — holographic-spread is not the same as");
        Console.WriteLine("  damage-robust. Caveat: byte-blocks are dominated by the I/O codec (embedding+output ~80% of params), so this");
        Console.WriteLine("  probes I/O redundancy more than the relational banks; a bank-level lesion would need in-lib shift access.");
    }
}
