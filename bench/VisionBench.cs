// Copyright (c) 2026 Dongyang Stephen Chen, trading as Evaluated Applications. All rights reserved.
// Source-available for study and verification; commercial use, redistribution, or derivatives require permission. See LICENSE.

using System.Text;
using PrismFormer;

namespace PrismFormer.Bench;

/// <summary>
/// A VISION probe for the char-level AlgFormer: can the algebraic model SEE and DRAW? Images are small ASCII rasters —
/// a natural fit, since the model already consumes printable-ASCII token sequences. Each 12x12 shape (procedurally drawn
/// at a RANDOM position/size, so held-out instances are unseen) is written as one sequence "<class>:<grid>=<class>", and
/// the model is trained as a plain char-LM over it — which teaches BOTH directions at once:
///   • GENERATION  — prompt "<class>:" → draw the 144-cell grid (the middle of the sequence)
///   • RECOGNITION — prompt "<grid>=" → name the class (the end of the sequence)
/// Reports held-out recognition accuracy per class, a generated sample per class, and a round-trip score (draw → recognise).
/// Self-contained (no download); toy scale. Usage: prismformer-bench --vision [--epochs E].
/// </summary>
public static class VisionBench
{
    const int G = 10;                                   // grid side (10x10 = 100 cells) — char-LM cost is ~O(cells²)/image, so keep it modest
    static readonly char[] Cls = { 'q', 'o', 'x', 'v', 'h', 'd' };
    static readonly string[] Name = { "square", "disc", "cross", "vline", "hline", "diag" };

    static char[] Render(int cls, int cx, int cy, int r)
    {
        var g = new char[G * G]; for (var i = 0; i < g.Length; i++) g[i] = '.';
        void Set(int x, int y) { if (x >= 0 && x < G && y >= 0 && y < G) g[y * G + x] = '#'; }
        switch (cls)
        {
            case 0: for (var t = -r; t <= r; t++) { Set(cx - r, cy + t); Set(cx + r, cy + t); Set(cx + t, cy - r); Set(cx + t, cy + r); } break;   // square outline
            case 1: for (var y = -r; y <= r; y++) for (var x = -r; x <= r; x++) if (x * x + y * y <= r * r) Set(cx + x, cy + y); break;             // filled disc
            case 2: for (var t = -r; t <= r; t++) { Set(cx + t, cy); Set(cx, cy + t); } break;                                                     // plus / cross
            case 3: for (var t = -r; t <= r; t++) Set(cx, cy + t); break;                                                                          // vertical line
            case 4: for (var t = -r; t <= r; t++) Set(cx + t, cy); break;                                                                          // horizontal line
            case 5: for (var t = -r; t <= r; t++) Set(cx + t, cy + t); break;                                                                      // diagonal
        }
        return g;
    }

    static void Draw(char[] g, string label)
    {
        Console.WriteLine($"  {label}");
        for (var y = 0; y < G; y++) Console.WriteLine("    " + new string(g, y * G, G));
    }

    public static void Run(int epochs = 24)
    {
        try { Console.OutputEncoding = Encoding.UTF8; } catch { }
        var rng = new Random(7);

        // ---- char vocab: pad + grid marks + separators + one letter per class ----
        var chars = new List<char> { ' ' }; var cid = new Dictionary<char, int> { [' '] = 0 };
        int Cid(char c) { if (cid.TryGetValue(c, out var i)) return i; i = cid.Count; cid[c] = i; chars.Add(c); return i; }
        foreach (var c in new[] { '.', '#', '=', ':' }) Cid(c);
        foreach (var c in Cls) Cid(c);
        var vocab = Math.Max(16, cid.Count + 4);
        double[] Seed(int w) => w < chars.Count ? PhasorCodec.Encode(chars[w].ToString()) : new double[PhasorCodec.Dim];

        // ---- procedural instances: random class/position/size, deduped ----
        const int perClass = 48;
        var insts = new List<(int cls, int cx, int cy, int r)>();
        for (var cls = 0; cls < Cls.Length; cls++)
        {
            var seen = new HashSet<(int, int, int)>();
            for (var guard = 0; seen.Count < perClass && guard < perClass * 40; guard++)
            {
                var r = 2 + rng.Next(4);                                   // radius 2..5
                var cx = r + rng.Next(G - 2 * r); var cy = r + rng.Next(G - 2 * r);
                if (seen.Add((cx, cy, r))) insts.Add((cls, cx, cy, r));
            }
        }

        // ---- stable train / held-out split by hash (held-out position+size combos are never trained) ----
        static int Bucket((int cls, int cx, int cy, int r) it) { uint h = 2166136261; foreach (var c in $"{it.cls}|{it.cx}|{it.cy}|{it.r}") { h ^= c; h *= 16777619; } return (int)(h % 100); }
        var train = insts.Where(x => Bucket(x) >= 20).ToList();
        var held = insts.Where(x => Bucket(x) < 20).ToList();

        // one sequence per instance, trained as a char-LM: "<class>:<grid>=<class>"
        string Seq((int cls, int cx, int cy, int r) it) => Cls[it.cls] + ":" + new string(Render(it.cls, it.cx, it.cy, it.r)) + "=" + Cls[it.cls];
        var pairs = new List<(int[] Ctx, int Target)>();
        foreach (var it in train) { var ids = Seq(it).Select(Cid).ToArray(); for (var i = 1; i < ids.Length; i++) pairs.Add((ids[..i], ids[i])); }

        var model = new AlgFormer(vocab, shifts: 32, layers: 3, maxContext: G * G + 8,
            dModel: PhasorCodec.Dim, frozenPrefix: PhasorCodec.FrozenReals, embedSeed: Seed, seed: 1);

        Console.WriteLine($"PrismFormer VISION probe   {DateTime.Now:yyyy-MM-dd HH:mm}");
        Console.WriteLine($"grid {G}x{G} · {Cls.Length} classes ({string.Join(", ", Name)}) · train {train.Count} / held-out {held.Count} instances · {pairs.Count:N0} char-LM pairs");
        Console.WriteLine($"model {model.ParamCount:N0} params (d={PhasorCodec.Dim}, S={model.Shifts}, L={model.Layers}) · epochs {epochs}\n");
        Console.WriteLine("what the shapes look like (one random training instance per class):");
        foreach (var cls in Enumerable.Range(0, Cls.Length)) { var it = train.First(x => x.cls == cls); Draw(Render(it.cls, it.cx, it.cy, it.r), Name[cls]); }
        Console.WriteLine();

        // ---- recognition accuracy: prompt "<grid>=" → predict the class letter ----
        double RecAcc(List<(int cls, int cx, int cy, int r)> set, int? onlyCls = null)
        {
            var s = set.Where(x => onlyCls is null || x.cls == onlyCls).ToList();
            if (s.Count == 0) return double.NaN;
            var ok = 0;
            foreach (var it in s) { var ctx = (new string(Render(it.cls, it.cx, it.cy, it.r)) + "=").Select(Cid).ToArray(); if (model.Predict(ctx) == Cid(Cls[it.cls])) ok++; }
            return ok / (double)s.Count;
        }

        var sw = System.Diagnostics.Stopwatch.StartNew();
        model.Train(pairs, epochs, batchSize: 256, baseLr: 1.5e-3, seed: 1, onEpoch: (ep, loss) =>
        {
            if (ep % 2 == 0 || ep == epochs) { Console.WriteLine($"  epoch {ep,3}/{epochs}  loss {loss:F3}  held-out recognition {RecAcc(held):P0}  ({sw.Elapsed.TotalSeconds:F0}s)"); Console.Out.Flush(); }
        });

        // ---- report ----
        Console.WriteLine("\nRECOGNITION — held-out accuracy per class (unseen positions/sizes):");
        foreach (var cls in Enumerable.Range(0, Cls.Length)) Console.WriteLine($"  {Name[cls],-8}  train {RecAcc(train, cls),5:P0}   held-out {RecAcc(held, cls),5:P0}");
        Console.WriteLine($"  {"OVERALL",-8}  train {RecAcc(train),5:P0}   held-out {RecAcc(held),5:P0}");

        // ---- generation: prompt "<class>:" → draw the grid; round-trip = draw then recognise ----
        Console.WriteLine("\nGENERATION — the model DRAWS each class from its name (greedy):");
        var rtOk = 0;
        foreach (var cls in Enumerable.Range(0, Cls.Length))
        {
            var prompt = (Cls[cls] + ":").Select(Cid).ToArray();
            var gen = model.Generate(prompt, G * G);
            var g = new char[G * G];
            for (var i = 0; i < g.Length; i++) { var c = i < gen.Length && gen[i] < chars.Count ? chars[gen[i]] : '?'; g[i] = c == '.' || c == '#' ? c : '?'; }
            Draw(g, $"asked for \"{Name[cls]}\"");
            var back = (new string(g) + "=").Select(Cid).ToArray();
            if (model.Predict(back) == Cid(Cls[cls])) rtOk++;
        }
        Console.WriteLine($"\nround-trip (draw → recognise as the intended class): {rtOk}/{Cls.Length}");
        Console.WriteLine($"\ndone · {sw.Elapsed.TotalSeconds:F0}s");
    }
}
