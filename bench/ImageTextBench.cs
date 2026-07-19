// Copyright (c) 2026 Dongyang Stephen Chen, trading as Evaluated Applications. All rights reserved.
// Source-available for study and verification; commercial use, redistribution, or derivatives require permission. See LICENSE.

using PrismFormer;

namespace PrismFormer.Bench;

/// <summary>
/// TEXT <-> IMAGE holographic codec (--imgtext). Can the phasor codec carry IMAGES the way it carries numbers/symbols,
/// and bind them to TEXT so you can go both ways? An image is encoded HOLOGRAPHICALLY: the face is the bundle (sum) of a
/// position phasor for every ON pixel, imgFace = Σ_on Encode("p{x}_{y}"). Three things are tested, no training:
///   1) IMAGE CODEC round-trip: decode imgFace back to pixels (per-position correlation, threshold) — is the image
///      DECODABLE from its face, the way a number is? (This is "generation from the representation".)
///   2) IMAGE -> TEXT: recognise a NOISY image by correlating its face against the class prototypes -> class label.
///   3) TEXT -> IMAGE: bind label<->image in one holographic memory M = Σ bind(labelFace, imgFace); unbind a label to
///      retrieve its imgFace, then decode to pixels. Plus a COMPOSITIONAL demo: bundle two shapes -> decode -> both appear.
/// Honest scope: holographic associative + superposition codec, CAPACITY-LIMITED (VSA crosstalk grows with ON pixels and
/// class count); it retrieves and composes stored prototypes, it does NOT learn to synthesise novel images.
/// </summary>
internal static class ImageTextBench
{
    const int G = 12;   // G x G binary images

    // ---- phasor ops on interleaved-real faces ----
    static double[] Bind(double[] a, double[] b)
    {
        var h = new double[a.Length];
        for (var c = 0; c < a.Length / 2; c++)
        { double ar = a[2*c], ai = a[2*c+1], br = b[2*c], bi = b[2*c+1]; h[2*c] = ar*br - ai*bi; h[2*c+1] = ar*bi + ai*br; }
        return h;
    }
    static double[] Conj(double[] a) { var h = (double[])a.Clone(); for (var c = 0; c < a.Length/2; c++) h[2*c+1] = -h[2*c+1]; return h; }
    static double Corr(double[] a, double[] b) { double s = 0; for (var i = 0; i < a.Length; i++) s += a[i]*b[i]; return s; }
    static void AddInto(double[] acc, double[] x) { for (var i = 0; i < acc.Length; i++) acc[i] += x[i]; }
    static double[] PosFace(int x, int y) => PhasorCodec.Encode($"px_{x}_{y}");

    // ---- 10 distinct procedural shapes ----
    static readonly (string name, Func<int, int, bool> on)[] Shapes =
    {
        ("frame",   (x, y) => x == 0 || x == G-1 || y == 0 || y == G-1),
        ("square",  (x, y) => (x >= 2 && x <= 9 && (y == 2 || y == 9)) || (y >= 2 && y <= 9 && (x == 2 || x == 9))),
        ("plus",    (x, y) => x == 5 || x == 6 || y == 5 || y == 6),
        ("ex",      (x, y) => Math.Abs(x - y) <= 1 || Math.Abs(x + y - (G-1)) <= 1),
        ("ring",    (x, y) => { var d = Math.Sqrt((x-5.5)*(x-5.5)+(y-5.5)*(y-5.5)); return d >= 3.3 && d <= 4.7; }),
        ("disc",    (x, y) => Math.Sqrt((x-5.5)*(x-5.5)+(y-5.5)*(y-5.5)) <= 3.1),
        ("diamond", (x, y) => { var d = Math.Abs(x-5.5)+Math.Abs(y-5.5); return d >= 3.5 && d <= 4.6; }),
        ("hbars",   (x, y) => y == 2 || y == 5 || y == 8),
        ("vbars",   (x, y) => x == 2 || x == 5 || x == 8),
        ("dots",    (x, y) => x % 3 == 1 && y % 3 == 1),
    };

    static bool[,] Draw(Func<int, int, bool> on) { var px = new bool[G, G]; for (var y = 0; y < G; y++) for (var x = 0; x < G; x++) px[y, x] = on(x, y); return px; }
    static int Count(bool[,] px) { var n = 0; foreach (var b in px) if (b) n++; return n; }

    static double[] Encode(bool[,] px) { var f = new double[PhasorCodec.Dim]; for (var y = 0; y < G; y++) for (var x = 0; x < G; x++) if (px[y, x]) AddInto(f, PosFace(x, y)); return f; }

    // decode a face -> pixels: a position is ON if its phasor correlates strongly with the face. Threshold at half the
    // self-correlation (Corr(PosFace,PosFace) = C = Dim/2), where an ON pixel sits at ~C + crosstalk and OFF at ~crosstalk.
    static bool[,] Decode(double[] face)
    {
        var thr = (PhasorCodec.Dim / 2) * 0.5;
        var px = new bool[G, G];
        for (var y = 0; y < G; y++) for (var x = 0; x < G; x++) px[y, x] = Corr(PosFace(x, y), face) > thr;
        return px;
    }

    static (double pix, double iou) Compare(bool[,] a, bool[,] b)
    {
        int same = 0, inter = 0, uni = 0;
        for (var y = 0; y < G; y++) for (var x = 0; x < G; x++)
        { if (a[y, x] == b[y, x]) same++; if (a[y, x] && b[y, x]) inter++; if (a[y, x] || b[y, x]) uni++; }
        return (same / (double)(G * G), uni == 0 ? 1 : inter / (double)uni);
    }

    static void RenderPair(string title, bool[,] left, bool[,] right, string lh, string rh)
    {
        Console.WriteLine($"  {title}");
        Console.WriteLine($"    {lh,-G}    {rh}");
        for (var y = 0; y < G; y++)
        {
            var l = ""; var r = "";
            for (var x = 0; x < G; x++) { l += left[y, x] ? "#" : "."; r += right[y, x] ? "#" : "."; }
            Console.WriteLine($"    {l}    {r}");
        }
    }

    public static void Run()
    {
        try { Console.OutputEncoding = System.Text.Encoding.UTF8; } catch { }
        Console.WriteLine($"TEXT <-> IMAGE holographic codec — encode images as phasor faces, bind to text, decode both ways (NO training)   {DateTime.Now:yyyy-MM-dd HH:mm}\n");

        var names = Shapes.Select(s => s.name).ToArray();
        var imgs = Shapes.Select(s => Draw(s.on)).ToArray();
        var faces = imgs.Select(Encode).ToArray();
        var labels = names.Select(n => PhasorCodec.Encode(n)).ToArray();
        Console.WriteLine($"  {names.Length} shapes on a {G}x{G} grid (ON pixels: {string.Join(", ", names.Zip(imgs, (n, im) => $"{n} {Count(im)}"))})\n");

        // ---- 1) IMAGE CODEC round-trip: decode the face back to pixels ----
        double pixSum = 0, iouSum = 0;
        for (var i = 0; i < faces.Length; i++) { var (p, io) = Compare(imgs[i], Decode(faces[i])); pixSum += p; iouSum += io; }
        Console.WriteLine("  1) IMAGE CODEC round-trip (encode -> face -> decode to pixels):");
        Console.WriteLine($"     mean pixel accuracy {pixSum / faces.Length:P1}   mean IoU {iouSum / faces.Length:P1}   (perfect = the face is losslessly decodable)\n");

        // ---- 2) IMAGE -> TEXT: recognise a NOISY image ----
        var rng = new Random(7);
        int okClean = 0, okNoisy = 0, trials = 0;
        foreach (var _ in Enumerable.Range(0, 25))
            for (var i = 0; i < imgs.Length; i++)
            {
                var noisy = (bool[,])imgs[i].Clone();
                for (var y = 0; y < G; y++) for (var x = 0; x < G; x++) if (rng.NextDouble() < 0.08) noisy[y, x] = !noisy[y, x];   // 8% pixel flips
                var q = Encode(noisy);
                var best = 0; var bv = double.NegativeInfinity;
                for (var j = 0; j < faces.Length; j++) { var c = Corr(q, faces[j]); if (c > bv) { bv = c; best = j; } }
                if (best == i) okNoisy++; trials++;
            }
        for (var i = 0; i < faces.Length; i++) { var best = 0; var bv = double.NegativeInfinity; for (var j = 0; j < faces.Length; j++) { var c = Corr(faces[i], faces[j]); if (c > bv) { bv = c; best = j; } } if (best == i) okClean++; }
        Console.WriteLine("  2) IMAGE -> TEXT (recognise image, output its class label):");
        Console.WriteLine($"     clean {okClean}/{faces.Length}   noisy (8% pixel flips) {okNoisy / (double)trials:P1} over {trials} trials\n");

        // ---- 3) TEXT -> IMAGE via one holographic memory M = Σ bind(label, image) ----
        var M = new double[PhasorCodec.Dim];
        for (var i = 0; i < faces.Length; i++) AddInto(M, Bind(labels[i], faces[i]));
        double tPix = 0, tIou = 0;
        for (var i = 0; i < labels.Length; i++)
        {
            var retrieved = Bind(M, Conj(labels[i]));   // unbind the (unitary) label -> imgFace_i + crosstalk from other classes
            var (p, io) = Compare(imgs[i], Decode(retrieved));
            tPix += p; tIou += io;
        }
        Console.WriteLine("  3) TEXT -> IMAGE (unbind label from holographic memory M -> decode to pixels):");
        Console.WriteLine($"     mean pixel accuracy {tPix / labels.Length:P1}   mean IoU {tIou / labels.Length:P1}   (lower than round-trip: adds N-1 classes of association crosstalk)\n");

        // ---- visual: a few text->image reconstructions ----
        Console.WriteLine("  SAMPLES — text label in, image out (true | generated from the codec):\n");
        foreach (var i in new[] { 0, 2, 4, 6 })
        {
            var gen = Decode(Bind(M, Conj(labels[i])));
            RenderPair($"\"{names[i]}\"", imgs[i], gen, "true", "generated");
            Console.WriteLine();
        }

        // ---- compositional: bundle two shapes -> decode -> both appear (novel combination, never stored together) ----
        var comp = new double[PhasorCodec.Dim]; AddInto(comp, faces[2]); AddInto(comp, faces[8]);   // plus + vbars
        var want = new bool[G, G]; for (var y = 0; y < G; y++) for (var x = 0; x < G; x++) want[y, x] = imgs[2][y, x] || imgs[8][y, x];
        RenderPair("COMPOSITIONAL  bundle(\"plus\", \"vbars\")  — a combination never stored", want, Decode(comp), "union", "generated");
        var (cp, ci) = Compare(want, Decode(comp));
        Console.WriteLine($"\n     compositional pixel accuracy {cp:P1}   IoU {ci:P1}\n");

        Console.WriteLine("  READ: round-trip pixel accuracy shows the image FACE is decodable (generation from the representation, like a");
        Console.WriteLine("  number decodes from its face). Recognition = image->text; unbinding M = text->image; bundling = compositional");
        Console.WriteLine("  scenes. All capacity-limited by VSA crosstalk (ON-pixel count, class count) — a mechanism, not a trained synthesiser.");
    }
}
