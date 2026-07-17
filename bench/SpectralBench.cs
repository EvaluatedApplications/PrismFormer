// Copyright (c) 2026 Dongyang Stephen Chen, trading as Evaluated Applications. All rights reserved.
// Source-available for study and verification; commercial use, redistribution, or derivatives require permission. See LICENSE.

using PrismFormer;

namespace PrismFormer.Bench;

/// <summary>
/// SPECTRAL PROOF-OF-CONCEPT (--spectral). No training. Tests how to encode a sound/vibration spectrum by
/// REUSING the number codec, and whether it gives a usable similarity geometry (similar spectra correlate high,
/// faults low). Compares the RAW number codec (tuned for integer orthogonality -> too brittle for Hz) against a
/// LOG-FREQUENCY tuned encoding (encode log2(f)*res, so a few Hz stays similar and octaves stay distinct — the
/// perceptually correct metric). The "verify kit for audio": if the tuned geometry separates with zero training,
/// the codec-reuse thesis holds and the model only has to learn the temporal pattern on top.
/// </summary>
internal static class SpectralBench
{
    const double Res = 4.0;   // log-frequency resolution: bandwidth ~ 1/Res octave

    static double[] Spec(Func<double, double[]> enc, params double[] freqs) => PhasorCodec.Bundle(freqs.Select(enc).ToArray());
    static double Cos(double[] a, double[] b)
    {
        var na = PhasorCodec.Correlate(a, a); var nb = PhasorCodec.Correlate(b, b);
        return (na <= 0 || nb <= 0) ? 0 : PhasorCodec.Correlate(a, b) / Math.Sqrt(na * nb);
    }

    static readonly Func<double, double[]> Raw = f => PhasorCodec.NumberFace(f);                    // number codec, raw Hz
    static readonly Func<double, double[]> Log = f => PhasorCodec.NumberFace(Math.Log2(f) * Res);   // tuned: log-frequency

    public static void Run()
    {
        try { Console.OutputEncoding = System.Text.Encoding.UTF8; } catch { }
        Console.WriteLine("SPECTRAL codec proof-of-concept — NO TRAINING, pure encoding geometry");
        Console.WriteLine("comparing RAW number codec (Hz) vs LOG-FREQUENCY tuned codec (log2(f)*4)\n");

        // ── 1. single-tone locality: cos(face(220), face(f)) — want smooth, noise-robust, octave-distinct ──
        Console.WriteLine("1. single-tone locality — cosine(face(220 Hz), face(f)):");
        Console.WriteLine($"     {"f (Hz)",7} {"ratio",6}   {"RAW",7}  {"LOG-TUNED",9}");
        foreach (var f in new double[] { 220, 221, 223, 227, 233, 247, 277, 311, 330, 440, 660, 880 })
            Console.WriteLine($"     {f,7:0} {f / 220.0,6:0.00}   {Cos(Raw(220), Raw(f)),7:0.000}  {Cos(Log(220), Log(f)),9:0.000}");

        // ── 2. machine signatures, cosine to NORMAL (note: additive faults share normal's energy) ──
        Console.WriteLine("\n2. machine signatures (bundle of harmonics), cosine to NORMAL:");
        Console.WriteLine($"     {"case",-24} {"RAW",7}  {"LOG-TUNED",9}");
        var cases = new (string name, double[] fs)[]
        {
            ("normal (identical)",      new double[]{60,120,180,240}),
            ("same machine, drifted",   new double[]{61,119,181,239}),
            ("same, load change",       new double[]{60,122,178,244}),
            ("BEARING FAULT (+hi f)",    new double[]{60,120,180,240,410,650}),
            ("IMBALANCE (subharm 30Hz)", new double[]{30,60,120,180,240}),
            ("different machine 50Hz",   new double[]{50,100,150,200}),
            ("unrelated pump",           new double[]{145,290,435}),
        };
        var nRaw = Spec(Raw, 60, 120, 180, 240); var nLog = Spec(Log, 60, 120, 180, 240);
        foreach (var (name, fs) in cases)
            Console.WriteLine($"     {name,-24} {Cos(nRaw, Spec(Raw, fs)),7:0.000}  {Cos(nLog, Spec(Log, fs)),9:0.000}");

        // ── 3. streaming detector with realistic +/-2 Hz measurement noise on the NORMAL frames ──
        Console.WriteLine("\n3. streaming detector (+/-2 Hz noise on normals) — flag if cos < threshold:");
        Detect(Raw, "RAW", 0.90);
        Detect(Log, "LOG-TUNED", 0.90);

        Console.WriteLine("\nread-out: RAW is too brittle for Hz (integer-orthogonality tuning). LOG-TUNED gives smooth,");
        Console.WriteLine("noise-robust, octave-aware similarity with ZERO training — that's the right codec for a sensor.");
        Console.WriteLine("(additive-fault detection still wants prediction-error, not cosine-to-template — a task-framing point.)");
    }

    static void Detect(Func<double, double[]> enc, string label, double thr)
    {
        var rng = new Random(1);
        double[] Jit(double[] fs) => fs.Select(f => f + (rng.NextDouble() - 0.5) * 4).ToArray();
        var normal = Spec(enc, 60, 120, 180, 240);
        var stream = new (string lbl, double[] fs)[]
        {
            ("normal", new double[]{60,120,180,240}), ("normal", new double[]{60,120,180,240}),
            ("normal", new double[]{60,120,180,240}), ("normal", new double[]{60,120,180,240}),
            ("FAULT ", new double[]{30,60,120,180,240}),     // subharmonic (looseness) — shifts the pattern
            ("normal", new double[]{60,120,180,240}),
            ("FAULT ", new double[]{50,100,150,200}),         // frequency drift to a different machine
            ("normal", new double[]{60,120,180,240}),
        };
        int correct = 0; var sb = new System.Text.StringBuilder();
        foreach (var (lbl, fs) in stream)
        {
            var c = Cos(normal, Spec(enc, Jit(fs)));
            bool flag = c < thr, fault = lbl.Trim() == "FAULT";
            if (flag == fault) correct++;
            sb.Append($"{c:0.00}{(flag ? "!" : " ")} ");
        }
        Console.WriteLine($"     {label,-10} [{sb.ToString().Trim()}]  ->  {correct}/{stream.Length} correct");
    }
}
