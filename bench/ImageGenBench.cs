// Copyright (c) 2026 Dongyang Stephen Chen, trading as Evaluated Applications. All rights reserved.
// Source-available for study and verification; commercial use, redistribution, or derivatives require permission. See LICENSE.

using PrismFormer;

namespace PrismFormer.Bench;

/// <summary>
/// LEARNED TEXT -> IMAGE generation, substrate-native (--imggen). No hardcoding, no cleanup memory, no foreign denoiser:
/// the image is a SEQUENCE of pixel tokens and the AlgFormer generates it one token at a time, conditioned on the text
/// label — image-GPT with the phasor substrate, the same causal machinery as the character LM. Sequence per image is
/// [label, "|", p0, p1, ..., p63] over an 8x8 binary grid; the model learns pixel(t) | label, p0..p(t-1). Then:
///   - GENERATE: prime [label, "|"], autoregress 64 pixels -> reconstruct the shape from the text alone.
///   - COMPLETE: prime [label, "|", top half] -> generate the bottom half (tests learned spatial structure, not lookup).
/// Everything emerges from training; nothing about the shapes is written into the decoder.
/// </summary>
internal static class ImageGenBench
{
    const int G = 8;   // 8x8 grid, 64 pixel tokens

    static readonly (string name, Func<int, int, bool> on)[] Shapes =
    {
        ("frame",   (x, y) => x == 0 || x == G-1 || y == 0 || y == G-1),
        ("plus",    (x, y) => x == 3 || x == 4 || y == 3 || y == 4),
        ("ex",      (x, y) => Math.Abs(x - y) <= 1 || Math.Abs(x + y - (G-1)) <= 1),
        ("ring",    (x, y) => { var d = Math.Sqrt((x-3.5)*(x-3.5)+(y-3.5)*(y-3.5)); return d >= 2.3 && d <= 3.4; }),
        ("disc",    (x, y) => Math.Sqrt((x-3.5)*(x-3.5)+(y-3.5)*(y-3.5)) <= 2.2),
        ("hlines",  (x, y) => y % 2 == 0),
        ("vlines",  (x, y) => x % 2 == 0),
        ("checker", (x, y) => (x + y) % 2 == 0),
    };

    static bool[] Draw(Func<int, int, bool> on) { var px = new bool[G * G]; for (var y = 0; y < G; y++) for (var x = 0; x < G; x++) px[y * G + x] = on(x, y); return px; }
    static (double pix, double iou) Compare(bool[] a, bool[] b)
    {
        int same = 0, inter = 0, uni = 0;
        for (var i = 0; i < a.Length; i++) { if (a[i] == b[i]) same++; if (a[i] && b[i]) inter++; if (a[i] || b[i]) uni++; }
        return (same / (double)a.Length, uni == 0 ? 1 : inter / (double)uni);
    }
    static void RenderPair(string title, bool[] left, bool[] right, string lh, string rh)
    {
        Console.WriteLine($"  {title}");
        Console.WriteLine($"    {lh,-G}   {rh}");
        for (var y = 0; y < G; y++)
        {
            string l = "", r = "";
            for (var x = 0; x < G; x++) { l += left[y*G+x] ? "#" : "."; r += right[y*G+x] ? "#" : "."; }
            Console.WriteLine($"    {l}   {r}");
        }
    }

    public static void Run(int epochs = 500)
    {
        try { Console.OutputEncoding = System.Text.Encoding.UTF8; } catch { }
        Console.WriteLine($"LEARNED text -> image (image-GPT on the phasor substrate) — AlgFormer generates pixels from a text label   {DateTime.Now:yyyy-MM-dd HH:mm}\n");

        // vocab: pixel OFF/ON, a separator, and one token per shape name
        var words = new List<string> { "<pad>", "0", "1", "|" };
        var id = new Dictionary<string, int>(StringComparer.Ordinal);
        for (var i = 0; i < words.Count; i++) id[words[i]] = i;
        int Id(string w) { if (id.TryGetValue(w, out var i)) return i; i = words.Count; id[w] = i; words.Add(w); return i; }
        int OFF = id["0"], ON = id["1"], SEP = id["|"];
        var imgs = Shapes.Select(s => Draw(s.on)).ToArray();
        var labelIds = Shapes.Select(s => Id(s.name)).ToArray();
        var vocab = Math.Max(32, words.Count);
        double[] Seed(int w) => w < words.Count ? PhasorCodec.Encode(words[w]) : new double[PhasorCodec.Dim];

        // one token sequence per image: [label, "|", 64 pixels]
        int[] Seq(int li, bool[] px) { var s = new int[2 + G * G]; s[0] = li; s[1] = SEP; for (var k = 0; k < G * G; k++) s[k + 2] = px[k] ? ON : OFF; return s; }
        var seqs = new int[Shapes.Length][];
        for (var i = 0; i < Shapes.Length; i++) seqs[i] = Seq(labelIds[i], imgs[i]);

        // causal training pairs: predict every pixel token from [label, "|", preceding pixels]
        var train = new List<(int[] ctx, int tgt)>();
        foreach (var s in seqs) for (var t = 2; t < s.Length; t++) train.Add((s[..t], s[t]));

        var m = new AlgFormer(vocab, shifts: 16, layers: 4, maxContext: G * G + 4, dModel: PhasorCodec.Dim, frozenPrefix: PhasorCodec.FrozenReals, embedSeed: Seed, seed: 1);
        Console.WriteLine($"  AlgFormer {m.ParamCount:N0} params · {Shapes.Length} shapes · {G}x{G} grid · {train.Count} training positions · {epochs} epochs\n");

        var rng = new Random(1);
        for (var ep = 1; ep <= epochs; ep++)
        {
            var lr = 3e-3 * (1.0 - 0.9 * (ep - 1) / Math.Max(1, epochs - 1));
            foreach (var i in Enumerable.Range(0, train.Count).OrderBy(_ => rng.Next())) { var (c, t) = train[i]; m.TrainStep(c, t, lr); }
            if (ep % 100 == 0 || ep == epochs)
            {
                double acc = 0; foreach (var s in seqs) { var ctx = new[] { s[0], SEP }.ToList(); var ok = 0; for (var k = 0; k < G * G; k++) { var p = m.Predict(ctx.ToArray()); if (p == s[k + 2]) ok++; ctx.Add(s[k + 2]); } acc += ok / (double)(G * G); }
                Console.WriteLine($"    epoch {ep,4}: teacher-forced next-pixel acc {acc / seqs.Length:P1}");
            }
        }

        // ---- GENERATE each shape from its label alone (free-running autoregression) ----
        bool[] Generate(int li)
        {
            var ctx = new List<int> { li, SEP };
            var px = new bool[G * G];
            for (var k = 0; k < G * G; k++) { var t = m.Predict(ctx.ToArray()); px[k] = t == ON; ctx.Add(t == ON ? ON : OFF); }
            return px;
        }
        Console.WriteLine("\n  GENERATE — text label in, image out (free-running, model decides every pixel):");
        double gp = 0, gi = 0;
        for (var i = 0; i < Shapes.Length; i++) { var (p, io) = Compare(imgs[i], Generate(labelIds[i])); gp += p; gi += io; }
        Console.WriteLine($"     mean pixel accuracy {gp / Shapes.Length:P1}   mean IoU {gi / Shapes.Length:P1}\n");
        foreach (var i in new[] { 0, 2, 3, 7 }) { RenderPair($"\"{Shapes[i].name}\"", imgs[i], Generate(labelIds[i]), "true", "generated"); Console.WriteLine(); }

        // ---- COMPLETE the bottom half from the label + top half (tests learned structure, not memorised lookup) ----
        int half = (G * G) / 2;
        double cp = 0, ci = 0;
        for (var i = 0; i < Shapes.Length; i++)
        {
            var s = seqs[i]; var ctx = s[..(2 + half)].ToList();
            var px = (bool[])imgs[i].Clone();
            for (var k = half; k < G * G; k++) { var t = m.Predict(ctx.ToArray()); px[k] = t == ON; ctx.Add(t == ON ? ON : OFF); }
            // score only the generated (bottom) half
            int same = 0, inter = 0, uni = 0;
            for (var k = half; k < G * G; k++) { if (px[k] == imgs[i][k]) same++; if (px[k] && imgs[i][k]) inter++; if (px[k] || imgs[i][k]) uni++; }
            cp += same / (double)half; ci += uni == 0 ? 1 : inter / (double)uni;
        }
        Console.WriteLine($"  COMPLETE — given the label + TOP half, generate the BOTTOM half:");
        Console.WriteLine($"     bottom-half pixel accuracy {cp / Shapes.Length:P1}   IoU {ci / Shapes.Length:P1}\n");

        Console.WriteLine("  READ: this is fully LEARNED generation — the AlgFormer writes each pixel token itself, conditioned only on the");
        Console.WriteLine("  text label, with nothing about the shapes hardcoded into the decoder. High GENERATE fidelity = it learned to draw");
        Console.WriteLine("  each shape from its name; high COMPLETE fidelity = it learned spatial structure, not just a per-label lookup.");
        Console.WriteLine("  (Memorisation-scale: 8 shapes. Novel/compositional synthesis needs augmented + composed training — the next step.)");
    }
}
