// Copyright (c) 2026 Dongyang Stephen Chen, trading as Evaluated Applications. All rights reserved.
// Source-available for study and verification; commercial use, redistribution, or derivatives require permission. See LICENSE.

using PrismFormer;

namespace PrismFormer.Bench;

/// <summary>
/// FREE PROTOTYPE LEARNING (--prototype). The user's idea: take an image SET of a similar thing and "do the combine
/// thing" — bundle (superpose) the holographic encodings of K examples into ONE class prototype, then classify new
/// images by correlation. No gradient descent anywhere: the fixed vision codec extracts features, bundling averages
/// examples into a concept. This measures the FEW-SHOT curve — how few examples give a usable prototype, and whether
/// superposition saturates or degrades as K grows (the VSA capacity limit seen in --assoc). If a handful of bundled
/// examples already classifies held-out shapes at random position/size, that is representation-learning for free.
/// </summary>
internal static class PrototypeBench
{
    const int G = 10, NCLS = 6;
    static readonly int Dim = PhasorCodec.Dim, C = PhasorCodec.Dim / 2;
    static readonly string[] Name = { "square", "disc", "cross", "vline", "hline", "diag" };
    static readonly double[] Ax = new double[C], Ay = new double[C];
    static PrototypeBench()
    {
        var r = new Random(0xBEEF); const double sigma = 0.6;
        double Gauss() { double u1 = 1 - r.NextDouble(), u2 = 1 - r.NextDouble(); return Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2); }
        for (var k = 0; k < C; k++) { Ax[k] = Gauss() * sigma; Ay[k] = Gauss() * sigma; }
    }

    static char[] Render(int cls, int cx, int cy, int rad)
    {
        var g = new char[G * G]; Array.Fill(g, '.');
        void Set(int x, int y) { if (x >= 0 && x < G && y >= 0 && y < G) g[y * G + x] = '#'; }
        switch (cls)
        {
            case 0: for (var t = -rad; t <= rad; t++) { Set(cx - rad, cy + t); Set(cx + rad, cy + t); Set(cx + t, cy - rad); Set(cx + t, cy + rad); } break;
            case 1: for (var y = -rad; y <= rad; y++) for (var x = -rad; x <= rad; x++) if (x * x + y * y <= rad * rad) Set(cx + x, cy + y); break;
            case 2: for (var t = -rad; t <= rad; t++) { Set(cx + t, cy); Set(cx, cy + t); } break;
            case 3: for (var t = -rad; t <= rad; t++) Set(cx, cy + t); break;
            case 4: for (var t = -rad; t <= rad; t++) Set(cx + t, cy); break;
            case 5: for (var t = -rad; t <= rad; t++) Set(cx + t, cy + t); break;
        }
        return g;
    }
    static double[] ShapeFace(char[] g)
    {
        double cx = 0, cy = 0; int n = 0;
        for (var y = 0; y < G; y++) for (var x = 0; x < G; x++) if (g[y * G + x] == '#') { cx += x; cy += y; n++; }
        if (n == 0) return new double[Dim];
        cx /= n; cy /= n;
        double rms = 0; for (var y = 0; y < G; y++) for (var x = 0; x < G; x++) if (g[y * G + x] == '#') { var ddx = x - cx; var ddy = y - cy; rms += ddx * ddx + ddy * ddy; }
        rms = Math.Sqrt(rms / n) + 1e-6; const double Scale = 3.0;
        var f = new double[Dim];
        for (var y = 0; y < G; y++) for (var x = 0; x < G; x++) if (g[y * G + x] == '#')
        { double dx = (x - cx) / rms * Scale, dy = (y - cy) / rms * Scale; for (var k = 0; k < C; k++) { var ph = dx * Ax[k] + dy * Ay[k]; f[2 * k] += Math.Cos(ph); f[2 * k + 1] += Math.Sin(ph); } }
        return Norm(f);
    }
    static double[] Norm(double[] v) { double s = 0; foreach (var x in v) s += x * x; s = Math.Sqrt(s) + 1e-12; var o = new double[v.Length]; for (var i = 0; i < v.Length; i++) o[i] = v[i] / s; return o; }
    static double Cos(double[] a, double[] b) { double s = 0; for (var i = 0; i < a.Length; i++) s += a[i] * b[i]; return s; }
    static (int cls, char[] g) Sample(Random rng) { int cls = rng.Next(NCLS), rad = rng.Next(2, 4), cx = rng.Next(rad + 1, G - rad - 1), cy = rng.Next(rad + 1, G - rad - 1); return (cls, Render(cls, cx, cy, rad)); }

    public static void Run()
    {
        try { Console.OutputEncoding = System.Text.Encoding.UTF8; } catch { }
        Console.WriteLine("FREE PROTOTYPE LEARNING — bundle K holographic image encodings per class into a prototype, classify by");
        Console.WriteLine("correlation. NO gradient descent. How few examples give a usable concept? (test: 900 held-out, random pos/size)\n");

        var testRng = new Random(123);
        var test = Enumerable.Range(0, 900).Select(_ => Sample(testRng)).ToArray();   // fixed held-out set for every K

        Console.WriteLine($"  {"examples/class (K)",20} {"held-out accuracy",18}");
        foreach (var K in new[] { 1, 2, 3, 5, 10, 30, 100, 300 })
        {
            var trainRng = new Random(50 + K);
            var proto = new double[NCLS][]; for (var c = 0; c < NCLS; c++) proto[c] = new double[Dim];
            var got = new int[NCLS];
            while (got.Any(x => x < K)) { var (cls, g) = Sample(trainRng); if (got[cls] >= K) continue; var f = ShapeFace(g); for (var k = 0; k < Dim; k++) proto[cls][k] += f[k]; got[cls]++; }
            for (var c = 0; c < NCLS; c++) proto[c] = Norm(proto[c]);

            var ok = 0;
            foreach (var (cls, g) in test)
            { var f = ShapeFace(g); int best = 0; double bs = double.NegativeInfinity; for (var c = 0; c < NCLS; c++) { var s = Cos(f, proto[c]); if (s > bs) { bs = s; best = c; } } if (best == cls) ok++; }
            Console.WriteLine($"  {K,20} {(double)ok / test.Length,17:P1}");
        }
        Console.WriteLine($"\n  chance = {1.0 / NCLS,4:P0}. If K=1 (ONE example per class) already beats chance and a handful nears the ceiling,");
        Console.WriteLine("  that is a usable classifier built with zero training — free few-shot concept formation by superposition.");
        Console.WriteLine("  The codec extracts features (fixed); bundling does the 'learning' (average examples into a prototype).");
    }
}
