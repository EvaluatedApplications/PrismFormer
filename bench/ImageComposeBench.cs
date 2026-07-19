// Copyright (c) 2026 Dongyang Stephen Chen, trading as Evaluated Applications. All rights reserved.
// Source-available for study and verification; commercial use, redistribution, or derivatives require permission. See LICENSE.

using PrismFormer;
using PrismFormer.Gpu;

namespace PrismFormer.Bench;

/// <summary>
/// COMPOSITIONAL text -> image (--imgcompose). A scene is TWO shapes side by side — shape A in the left 6x6 slot, shape
/// B in the right — and the text is the two labels in order (order picks the slot). The AlgFormer is trained image-GPT
/// style (causal pixel tokens, nothing hardcoded) on a SUBSET of the 16 pairings and tested on the HELD-OUT pairings it
/// never saw. To draw a novel pair it must compose: which shape each label names, and which slot each text position maps
/// to, learned separately then combined. High held-out fidelity = real compositional generation; low = memorised pairings.
/// BIGGER model + GPU (auto, via GpuTrainer) + convergence logging so we can confirm it FITS train before reading held-out.
/// </summary>
internal static class ImageComposeBench
{
    const int S = 6, R = 6, C = 12;   // 6x6 stamps; scene 6 rows x 12 cols (two slots)

    static readonly (string name, Func<int, int, bool> on)[] Shapes =
    {
        ("box",   (x, y) => x == 0 || x == S-1 || y == 0 || y == S-1),
        ("cross", (x, y) => x == 2 || x == 3 || y == 2 || y == 3),
        ("ring",  (x, y) => { var d = Math.Sqrt((x-2.5)*(x-2.5)+(y-2.5)*(y-2.5)); return d >= 1.4 && d <= 2.5; }),
        ("ex",    (x, y) => Math.Abs(x - y) <= 0.5 || Math.Abs(x + y - (S-1)) <= 0.5),
    };

    static bool[] Stamp(Func<int, int, bool> on) { var p = new bool[S * S]; for (var y = 0; y < S; y++) for (var x = 0; x < S; x++) p[y * S + x] = on(x, y); return p; }
    static bool[] Scene(bool[] a, bool[] b) { var s = new bool[R * C]; for (var r = 0; r < R; r++) for (var c = 0; c < C; c++) s[r * C + c] = c < S ? a[r * S + c] : b[r * S + (c - S)]; return s; }
    static bool[] Half(bool[] scene, bool right) { var h = new bool[S * S]; for (var r = 0; r < R; r++) for (var c = 0; c < S; c++) h[r * S + c] = scene[r * C + (right ? c + S : c)]; return h; }
    static (double pix, double iou) Cmp(bool[] a, bool[] b)
    {
        int same = 0, inter = 0, uni = 0;
        for (var i = 0; i < a.Length; i++) { if (a[i] == b[i]) same++; if (a[i] && b[i]) inter++; if (a[i] || b[i]) uni++; }
        return (same / (double)a.Length, uni == 0 ? 1 : inter / (double)uni);
    }
    static void Render(string title, bool[] scene)
    {
        Console.WriteLine($"    {title}");
        for (var r = 0; r < R; r++) { var l = ""; for (var c = 0; c < C; c++) l += scene[r * C + c] ? "#" : "."; Console.WriteLine($"      {l}"); }
    }

    public static void Run(int epochs = 2500)
    {
        try { Console.OutputEncoding = System.Text.Encoding.UTF8; } catch { }
        Console.WriteLine($"COMPOSITIONAL text -> image (bigger model + GPU) — HOLD OUT unseen pairings   {DateTime.Now:yyyy-MM-dd HH:mm}\n");

        var words = new List<string> { "<pad>", "0", "1", "|" };
        var id = new Dictionary<string, int>(StringComparer.Ordinal);
        for (var i = 0; i < words.Count; i++) id[words[i]] = i;
        int Id(string w) { if (id.TryGetValue(w, out var i)) return i; i = words.Count; id[w] = i; words.Add(w); return i; }
        int OFF = id["0"], ON = id["1"], SEP = id["|"];
        var stamps = Shapes.Select(s => Stamp(s.on)).ToArray();
        var lab = Shapes.Select(s => Id(s.name)).ToArray();
        var vocab = Math.Max(32, words.Count);
        double[] Seed(int w) => w < words.Count ? PhasorCodec.Encode(words[w]) : new double[PhasorCodec.Dim];

        var held = new HashSet<(int, int)> { (0, 1), (1, 2), (2, 3), (3, 0) };   // 4-cycle: every shape still appears in both slots in training
        var all = new List<(int a, int b)>();
        for (var a = 0; a < 4; a++) for (var b = 0; b < 4; b++) all.Add((a, b));
        var trainCombos = all.Where(p => !held.Contains(p)).ToList();

        int[] Seq(int a, int b) { var sc = Scene(stamps[a], stamps[b]); var s = new int[3 + R * C]; s[0] = lab[a]; s[1] = lab[b]; s[2] = SEP; for (var k = 0; k < R * C; k++) s[k + 3] = sc[k] ? ON : OFF; return s; }
        var train = new List<(int[] ctx, int tgt)>();
        foreach (var (a, b) in trainCombos) { var s = Seq(a, b); for (var t = 3; t < s.Length; t++) train.Add((s[..t], s[t])); }

        // BIGGER model: shifts 64, layers 6 (was 16/4). dModel fixed at the phasor width.
        var m = new AlgFormer(vocab, shifts: 64, layers: 6, maxContext: R * C + 6, dModel: PhasorCodec.Dim, frozenPrefix: PhasorCodec.FrozenReals, embedSeed: Seed, seed: 1);
        Console.WriteLine($"  AlgFormer {m.ParamCount:N0} params · 4 shapes · scene {R}x{C} · train {trainCombos.Count} / hold out {held.Count} · {train.Count} positions · {epochs} epochs");

        GpuTrainer? gpu = null;
        try { if (GpuDevice.HasGpu) { gpu = new GpuTrainer(m, tokenBudget: 262144); Console.WriteLine($"  device: {GpuDevice.Describe}\n"); } else Console.WriteLine($"  device: {GpuDevice.Describe} — training on CPU\n"); }
        catch (Exception e) { Console.WriteLine("  GPU init failed -> CPU: " + e.Message.Split('\n')[0] + "\n"); gpu = null; }

        double TeacherForced()
        {
            double acc = 0;
            foreach (var (a, b) in trainCombos) { var s = Seq(a, b); var ok = 0; for (var t = 3; t < s.Length; t++) if (m.Predict(s[..t]) == s[t]) ok++; acc += ok / (double)(s.Length - 3); }
            return acc / trainCombos.Count;
        }

        const int BATCH = 4096;   // whole training set in one GPU call (short sequences, so it fits the token budget)
        var rng = new Random(1);
        var stable = 0;
        for (var ep = 1; ep <= epochs; ep++)
        {
            var lr = 3e-3 * (1.0 - 0.9 * (ep - 1) / Math.Max(1, epochs - 1));
            var order = Enumerable.Range(0, train.Count).OrderBy(_ => rng.Next()).ToArray();
            if (gpu != null)
                for (var b = 0; b < order.Length; b += BATCH)
                {
                    var batch = new List<(int[] Ctx, int Target)>();
                    for (var i = b; i < Math.Min(b + BATCH, order.Length); i++) batch.Add(train[order[i]]);
                    gpu.TrainBatch(batch, lr);
                }
            else
                foreach (var i in order) { var (c, t) = train[i]; m.TrainStep(c, t, lr); }

            if (ep % 50 == 0 || ep == epochs)
            {
                var tf = TeacherForced();
                Console.WriteLine($"    epoch {ep,4}: teacher-forced next-pixel acc {tf:P1}");
                if (tf >= 0.9999) { if (++stable >= 2) { Console.WriteLine($"    -> converged; stopping early at epoch {ep}"); break; } }
                else stable = 0;
            }
        }
        gpu?.Dispose();

        bool[] Gen(int a, int b) { var ctx = new List<int> { lab[a], lab[b], SEP }; var sc = new bool[R * C]; for (var k = 0; k < R * C; k++) { var t = m.Predict(ctx.ToArray()); sc[k] = t == ON; ctx.Add(t == ON ? ON : OFF); } return sc; }
        (double lp, double li, double rp, double ri) Score(int a, int b) { var g = Gen(a, b); var (lp, li) = Cmp(stamps[a], Half(g, false)); var (rp, ri) = Cmp(stamps[b], Half(g, true)); return (lp, li, rp, ri); }

        double slp = 0, sli = 0, srp = 0, sri = 0;
        foreach (var (a, b) in trainCombos) { var (lp, li, rp, ri) = Score(a, b); slp += lp; sli += li; srp += rp; sri += ri; }
        var ns = trainCombos.Count;
        Console.WriteLine($"\n  SEEN pairings (should reproduce): left pixel {slp/ns:P0} IoU {sli/ns:P0}   right pixel {srp/ns:P0} IoU {sri/ns:P0}");

        double hlp = 0, hli = 0, hrp = 0, hri = 0;
        foreach (var (a, b) in held) { var (lp, li, rp, ri) = Score(a, b); hlp += lp; hli += li; hrp += rp; hri += ri; }
        var nh = held.Count;
        Console.WriteLine($"  HELD-OUT (NOVEL pairings): left pixel {hlp/nh:P0} IoU {hli/nh:P0}   right pixel {hrp/nh:P0} IoU {hri/nh:P0}\n");

        foreach (var (a, b) in held)
        {
            Console.WriteLine($"  \"{Shapes[a].name} | {Shapes[b].name}\"  (held out):");
            Render("true:", Scene(stamps[a], stamps[b]));
            Render("generated:", Gen(a, b));
            Console.WriteLine();
        }

        Console.WriteLine("  READ: first confirm SEEN reproduces (~100%) — that means the model FIT the training scenes. Only then does HELD-OUT");
        Console.WriteLine("  mean anything: high = it COMPOSED a pairing it never saw (real generation); low = it memorised the pairings it saw.");
        GpuDevice.Shutdown();
    }
}
