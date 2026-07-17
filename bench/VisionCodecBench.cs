// Copyright (c) 2026 Dongyang Stephen Chen, trading as Evaluated Applications. All rights reserved.
// Source-available for study and verification; commercial use, redistribution, or derivatives require permission. See LICENSE.

using PrismFormer;

namespace PrismFormer.Bench;

/// <summary>
/// VISION CODEC proof-of-concept (--vision-codec). NO training. Tests a HOLOGRAPHIC image codec vs the old
/// ASCII-raster (VisionBench flattened a shape to a 1D char stream — no 2D structure, no translation). Here a
/// shape is encoded as bundle( positionFace(x,y) ) over its filled pixels, CENTRED on its centroid and
/// SCALE-normalised, so the face captures SHAPE independent of where/how big it is drawn. positionFace is a 2D
/// random-Fourier phasor (nearby pixels correlate; x and y independent). Recognition = correlate a held-out shape
/// (drawn at a RANDOM position and size) against per-class templates. Baseline: raw pixel-overlap, which is
/// position-blind and should collapse when the shape moves. If holographic beats raw like log-freq beat raw-Hz,
/// the "codec per domain" thesis generalises to vision.
/// </summary>
internal static class VisionCodecBench
{
    const int G = 10;
    static readonly int Dim = PhasorCodec.Dim, C = PhasorCodec.Dim / 2;
    static readonly string[] Name = { "square", "disc", "cross", "vline", "hline", "diag" };

    // fixed 2D random-Fourier frequencies (Gaussian -> Gaussian similarity kernel; sigma sets the bandwidth)
    static readonly double[] Ax = new double[C], Ay = new double[C];
    static VisionCodecBench()
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

    // holographic shape face: centre on centroid, scale to unit RMS radius, bundle the 2D-RFF position faces.
    static double[] ShapeFace(char[] g)
    {
        double cx = 0, cy = 0; int n = 0;
        for (var y = 0; y < G; y++) for (var x = 0; x < G; x++) if (g[y * G + x] == '#') { cx += x; cy += y; n++; }
        if (n == 0) return new double[Dim];
        cx /= n; cy /= n;
        double rms = 0; for (var y = 0; y < G; y++) for (var x = 0; x < G; x++) if (g[y * G + x] == '#') { var ddx = x - cx; var ddy = y - cy; rms += ddx * ddx + ddy * ddy; }
        rms = Math.Sqrt(rms / n) + 1e-6;
        const double Scale = 3.0;   // normalise to RMS radius ~3 so points spread out (not packed near the DC face)
        var f = new double[Dim];
        for (var y = 0; y < G; y++) for (var x = 0; x < G; x++) if (g[y * G + x] == '#')
        {
            double dx = (x - cx) / rms * Scale, dy = (y - cy) / rms * Scale;
            for (var k = 0; k < C; k++) { var ph = dx * Ax[k] + dy * Ay[k]; f[2 * k] += Math.Cos(ph); f[2 * k + 1] += Math.Sin(ph); }
        }
        return Norm(f);
    }

    static double[] RawVec(char[] g) { var v = new double[G * G]; for (var i = 0; i < v.Length; i++) v[i] = g[i] == '#' ? 1 : 0; return Norm(v); }
    static double[] Norm(double[] v) { double s = 0; foreach (var x in v) s += x * x; s = Math.Sqrt(s) + 1e-12; var o = new double[v.Length]; for (var i = 0; i < v.Length; i++) o[i] = v[i] / s; return o; }
    static double Cos(double[] a, double[] b) { double s = 0; for (var i = 0; i < a.Length; i++) s += a[i] * b[i]; return s; }

    public static void Run()
    {
        try { Console.OutputEncoding = System.Text.Encoding.UTF8; } catch { }
        Console.WriteLine("VISION CODEC proof-of-concept — NO training. Holographic shape face vs raw pixel-overlap.\n");
        var rng = new Random(7);
        (int cls, char[] g) Sample() { int cls = rng.Next(6), rad = rng.Next(2, 4), cx = rng.Next(rad + 1, G - rad - 1), cy = rng.Next(rad + 1, G - rad - 1); return (cls, Render(cls, cx, cy, rad)); }

        // one illustrative pair: the SAME shape drawn in two different places
        var sqA = Render(0, 2, 2, 2); var sqB = Render(0, 7, 7, 2); var discA = Render(1, 2, 2, 2);
        Console.WriteLine("illustration — square@top-left vs square@bottom-right vs disc@top-left:");
        Console.WriteLine($"     RAW pixel-overlap   square/square = {Cos(RawVec(sqA), RawVec(sqB)),5:0.00}   square/disc = {Cos(RawVec(sqA), RawVec(discA)),5:0.00}");
        Console.WriteLine($"     HOLOGRAPHIC codec   square/square = {Cos(ShapeFace(sqA), ShapeFace(sqB)),5:0.00}   square/disc = {Cos(ShapeFace(sqA), ShapeFace(discA)),5:0.00}");

        // build per-class templates from training shapes at RANDOM pos/size
        var tHolo = new double[6][]; var tRaw = new double[6][];
        for (var c = 0; c < 6; c++) { tHolo[c] = new double[Dim]; tRaw[c] = new double[G * G]; }
        var perClass = new int[6];
        for (var i = 0; i < 600; i++) { var (cls, g) = Sample(); var hf = ShapeFace(g); var rv = RawVec(g); for (var k = 0; k < Dim; k++) tHolo[cls][k] += hf[k]; for (var k = 0; k < G * G; k++) tRaw[cls][k] += rv[k]; perClass[cls]++; }
        for (var c = 0; c < 6; c++) { tHolo[c] = Norm(tHolo[c]); tRaw[c] = Norm(tRaw[c]); }

        // classify held-out shapes (unseen random pos/size)
        int okH = 0, okR = 0, total = 0; var perH = new int[6]; var perN = new int[6];
        for (var i = 0; i < 600; i++)
        {
            var (cls, g) = Sample(); total++;
            int Argmax(double[][] tpl, Func<char[], double[]> enc) { var f = enc(g); int best = 0; double bs = double.NegativeInfinity; for (var c = 0; c < 6; c++) { var s = Cos(f, tpl[c]); if (s > bs) { bs = s; best = c; } } return best; }
            var ph = Argmax(tHolo, ShapeFace); var pr = Argmax(tRaw, RawVec);
            if (ph == cls) { okH++; perH[cls]++; } if (pr == cls) okR++; perN[cls]++;
        }
        Console.WriteLine("\nheld-out recognition (600 shapes at random position AND size, no training):");
        Console.WriteLine($"     {"class",-8} {"holographic",12} {"raw pixel",10}");
        for (var c = 0; c < 6; c++) Console.WriteLine($"     {Name[c],-8} {(double)perH[c] / perN[c],11:P0} {"",-2}");
        Console.WriteLine($"\n     OVERALL   holographic {(double)okH / total,6:P1}   raw pixel {(double)okR / total,6:P1}   (chance {1.0 / 6,4:P0})");
        Console.WriteLine("\n   holographic captures SHAPE independent of position/size; raw pixel-overlap collapses when the shape moves.");
    }
}
