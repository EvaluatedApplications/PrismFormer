// Copyright (c) 2026 Dongyang Stephen Chen, trading as Evaluated Applications. All rights reserved.
// Source-available for study and verification; commercial use, redistribution, or derivatives require permission. See LICENSE.

using PrismFormer;
using PrismFormer.Bench;

// ═══════════════════════════════════════════════════════════════════════════════════════════════════
//  PrismFormer vs a POUND-FOR-POUND transformer — a broad, varied multi-task benchmark with a stable
//  TRAIN vs HELD-OUT split. Both models share the vocab, context, train examples (same order), LR schedule
//  and parameter budget (the transformer's shape is auto-searched to match PrismFormer's params). Tasks span:
//    • COMPUTE  (should GENERALISE): add, sub, mul, div, seq, gt, max, min, parity
//    • ALGORITHMIC (generalises via routing): copy
//    • FACT RECALL (should MEMORISE, not generalise): capital, antonym, category
//  Held-out operands/instances are never trained, so held-out accuracy on COMPUTE is the generalisation test.
//  Usage: prismformer-bench [--epochs E] [--hi N]
// ═══════════════════════════════════════════════════════════════════════════════════════════════════

try { Console.OutputEncoding = System.Text.Encoding.UTF8; } catch { }
var epochs = 150; var hi = 12;
for (var i = 0; i < args.Length - 1; i++)
{
    if (args[i] == "--epochs") epochs = int.Parse(args[i + 1]);
    if (args[i] == "--hi") hi = int.Parse(args[i + 1]);
}
// --tuned-baseline: swap the transformer baseline for the MODERN, properly-tuned recipe (pre-norm LayerNorm +
// linear-warmup→cosine LR + tuned Adam). It only strengthens the baseline (AlgFormer is untouched); combine with
// the multi-task default, --lm, or --codec-baseline. Run each headline task once with and once without the flag.
var tuned = args.Contains("--tuned-baseline");
if (args.Contains("--sample")) { UpgradeBench.RunSample(args.SkipWhile(a => a != "--sample").Skip(1).ToArray()); return; }   // load checkpoint(s) + greedily generate — trained (coherent) vs fresh (garbage)
if (args.Contains("--gradcheck")) { UpgradeBench.RunGradcheck(); return; }   // bitwise gradient checksum — must be unchanged by the scratch-reuse refactor
if (args.Contains("--profile")) { UpgradeBench.RunProfile(); return; }   // training throughput/CPU-utilization/GC profiler (why the CPU doesn't saturate)
if (args.Contains("--spec")) { UpgradeBench.RunSpec(); return; }   // verify a c256 checkpoint LoadUpgrades into the current spec byte-clean (safe to ship the bump)
if (args.Contains("--upgrade-lm")) { UpgradeBench.RunLm(); return; }   // BENEFIT: train each upgrade config on char LM next-token, held-out accuracy
if (args.Contains("--upgrade")) { UpgradeBench.Run(); return; }   // perf cost of the expand-in-place upgrades (grow Shifts / grow Context)
if (args.Contains("--vision")) { VisionBench.Run(epochs == 150 ? 24 : epochs); return; }   // isolated capability: ASCII-raster recognition + generation
if (args.Contains("--spectral")) { SpectralBench.Run(); return; }   // proof-of-concept: encode a sound/vibration spectrum via the number codec, test similarity/anomaly geometry (no training)
if (args.Contains("--spectral-seq")) { SpectralSeqBench.Run(); return; }   // spectral PrismFormer: train on normal machine rhythm (log-freq tokens), detect anomalies by prediction surprise
if (args.Contains("--vision-codec")) { VisionCodecBench.Run(); return; }   // holographic image codec: shape = bundle(positionFace), position-invariant recognition by correlation (no training)
if (args.Contains("--hash")) { HashBench.Run(); return; }   // hash learnability: held-out generalisation collapses as diffusion increases (memorise vs learn)
if (args.Contains("--assoc")) { AssocBench.Run(); return; }   // disorder-codec associative memory: reverse hash-lookup of STORED pairs (a rainbow table), capacity-limited, 0% unseen
if (args.Contains("--crack")) { HashCrackBench.Run(); return; }   // train to invert small permutation hashes; measure held-out LOSS vs chance (how close it gets)
if (args.Contains("--crack-faces")) { CodecCrackBench.Run(); return; }   // sweep structured/disordered x frozen/unfrozen codec on hash inversion; held-out loss vs chance
if (args.Contains("--lesion")) { LesionBench.Run(); return; }   // Levin-inspired basal competency: lesion tolerance (graceful degradation) + regeneration (heal after damage)
if (args.Contains("--colony")) { ColonyBench.Run(); return; }   // Levin-inspired COLLECTIVE competency: a federated colony reaches the goal despite nodes failing every round (broken-hardware analog)
if (args.Contains("--diverse")) { DiverseBench.Run(); return; }   // Levin-inspired heterogeneous collective: diverse-config ensemble vs homogeneous (multiple algorithms grouped)
if (args.Contains("--plasticity")) { PlasticityBench.Run(); return; }   // Levin-inspired plasticity/degeneracy: amputate the critical region, hold it dead, see if the function relocates
if (args.Contains("--emergence")) { EmergenceBench.Run(); return; }   // Levin-inspired EMERGENCE: the collective learns the addition rule (solves pairs no node ever saw) — competency in the group, in no member
if (args.Contains("--mesh")) { MeshBench.Run(); return; }   // FAITHFUL Prism Studio mesh: autonomous models chatter via weight-slice elastic-averaging + pair-gossip (NO gradient summing)
if (args.Contains("--collapse")) { CollapseBench.Run(); return; }   // does the bleed damage holographic info? frozen vs unfrozen codec, algebra accuracy per tick — tests if the codec-pinning prevents collapse
if (args.Contains("--average")) { XferBench.Run(); return; }   // can you average SEPARATELY-trained models? same/diff init x frozen/no codec — does the codec let genuinely-independent models average?
if (args.Contains("--swarmdemo")) { SwarmDemoBench.Run(args); return; }   // clean before/after demo of the two swarm mechanisms: (1) bit-exact master–slave gradient sharing, (2) elastic-averaging skill bleed
if (args.Contains("--codec-baseline")) { BaselineControlBench.Run(epochs == 150 ? 800 : epochs, tuned: tuned); return; }   // paper1 §6-A control: seed the transformer baseline from the same codec — is the gap init or architecture? (train long enough that the transformer FITS train)
if (args.Contains("--imgcompose")) { ImageComposeBench.Run(); return; }   // COMPOSITIONAL text->image: draw two shapes from two labels; held-out pairings test NOVEL synthesis vs memorisation
if (args.Contains("--imggen")) { ImageGenBench.Run(); return; }   // LEARNED text->image: AlgFormer generates an image pixel-by-pixel from a text label (image-GPT on the phasor substrate, nothing hardcoded)
if (args.Contains("--imgtext")) { ImageTextBench.Run(); return; }   // text<->image holographic codec: encode images as phasor faces, bind to text, decode both ways (round-trip, recognise, generate, compose)
if (args.Contains("--revinfer")) { RevInferBench.Run(); return; }   // reverse inference: beat the paper's 0% via scratchpad (fact-in-context select) and algebra (commutative-bind HRR memory, reverse for free)
if (args.Contains("--collatz")) { CollatzBench.Run(); return; }   // Collatz stopping-time probe: how much of a chaotic sequence can PrismFormer learn? (trend vs spikes, held-out + extrapolate)
if (args.Contains("--prototype")) { PrototypeBench.Run(); return; }   // free prototype learning: bundle K holographic image encodings per class into a concept, few-shot curve, no training
if (args.Contains("--distinguish")) { CryptoDistinguisherBench.Run(); return; }   // reduced-round neural distinguisher (Gohr-style): PrismFormer's reach at telling round-reduced output from random
if (args.Contains("--speck")) { SpeckDistinguisherBench.Run(); return; }   // PrismFormer vs Gohr's real Speck32/64 benchmark: distinguisher accuracy at 5-8 rounds
if (args.Contains("--inspect")) { ResearchInspect.Run(); return; }   // targeted isolated addition, multi-seed averages, + face inspection (decode the model's internals) vs a transformer
if (args.Contains("--columnar")) { ColumnarBench.Run(); return; }   // end-to-end columnar addition: length extrapolation + per-column face inspection vs a transformer
if (args.Contains("--extrap")) { ExtrapolationBench.Run(); return; }   // isolated capability: out-of-range magnitude extrapolation
if (args.Contains("--lm")) { LanguageBench.Run(tuned: tuned); return; }             // isolated capability: character language modelling
if (args.Contains("--scale")) { ScaleBench.Run(); return; }            // does it scale? held-out compute vs model size
if (!args.Contains("--legacy")) { MultiTaskBench.Run(seeds: 5, epochs: epochs, tuned: tuned); return; }   // seeded multi-task generalisation (train+held mean±sd) — the paper's §4.2
var rng = new Random(7);
var all = new List<Inst>();
void Emit(string task, string[] prompt, object target) => all.Add(new Inst(task, prompt, target.ToString()!));

// ---- COMPUTE tasks (generalise to unseen operands) ----
for (var a = 0; a <= hi; a++)
    for (var b = 0; b <= hi; b++)
    {
        Emit("add", new[] { $"{a}", "+", $"{b}", "=" }, a + b);
        if (a >= b) Emit("sub", new[] { $"{a}", "-", $"{b}", "=" }, a - b);
        if (a >= 1 && b >= 1) { Emit("mul", new[] { $"{a}", "*", $"{b}", "=" }, a * b); Emit("div", new[] { $"{a * b}", "/", $"{b}", "=" }, a); }
        Emit("gt", new[] { $"{a}", ">", $"{b}", "=" }, a > b ? "yes" : "no");
        Emit("max", new[] { "max", $"{a}", $"{b}", "=" }, Math.Max(a, b));
        Emit("min", new[] { "min", $"{a}", $"{b}", "=" }, Math.Min(a, b));
    }
for (var start = 0; start <= 10; start++)
    for (var step = 1; step <= 4; step++)
        Emit("seq", new[] { $"{start}", $"{start + step}", $"{start + 2 * step}", "=" }, start + 3 * step);
for (var n = 0; n <= 49; n++)
    Emit("parity", new[] { $"{n}", "is", "=" }, n % 2 == 0 ? "even" : "odd");

// ---- ALGORITHMIC (copy the first of three — attention control) ----
string[] W = { "red", "blue", "green", "dog", "cat", "sun", "moon", "tree", "fish", "gold", "star", "leaf" };
var seenTriple = new HashSet<string>();
for (var n = 0; n < 260 && seenTriple.Count < 220; n++)
{
    string w0 = W[rng.Next(W.Length)], w1 = W[rng.Next(W.Length)], w2 = W[rng.Next(W.Length)];
    if (!seenTriple.Add($"{w0} {w1} {w2}")) continue;
    Emit("copy", new[] { "first", "of", w0, w1, w2, ":" }, w0);
}

// ---- RELATIONAL facts with an INFERABLE held-out. The forward fact is trained; the reverse/symmetric form is held
//      out. A held-out item is therefore answerable from what was trained, exactly as a person would reason — testing
//      whether the model learned the RELATION rather than only the memorised pair. (Arbitrary category membership is
//      omitted: with hashed word faces it is not inferable, so held-out there would be an unfair 0.)
var capitals = new[] { ("france", "paris"), ("england", "london"), ("germany", "berlin"), ("japan", "tokyo"), ("spain", "madrid"), ("italy", "rome"), ("portugal", "lisbon"), ("austria", "vienna"), ("greece", "athens"), ("norway", "oslo"), ("ireland", "dublin"), ("egypt", "cairo"), ("canada", "ottawa"), ("russia", "moscow"), ("china", "beijing"), ("thailand", "bangkok"), ("cuba", "havana"), ("peru", "lima"), ("poland", "warsaw"), ("sweden", "stockholm"), ("finland", "helsinki"), ("hungary", "budapest"), ("denmark", "copenhagen"), ("turkey", "ankara") };
var antonyms = new[] { ("up", "down"), ("left", "right"), ("day", "night"), ("open", "shut"), ("win", "lose"), ("buy", "sell"), ("push", "pull"), ("give", "take"), ("love", "hate"), ("war", "peace"), ("north", "south"), ("east", "west"), ("king", "queen"), ("front", "back"), ("top", "bottom"), ("friend", "enemy"), ("rise", "fall"), ("accept", "reject"), ("arrive", "depart"), ("remember", "forget"), ("enter", "exit"), ("sink", "float"), ("laugh", "cry"), ("import", "export") };

var computeTasks = new[] { "add", "sub", "mul", "div", "seq", "gt", "max", "min", "parity" };
var relationalTasks = new[] { "antonym", "capital" };
var tasks = computeTasks.Append("copy").Concat(relationalTasks).ToArray();

// stable train / held-out split for the hash-split tasks (compute + copy)
static int Bucket(Inst x) { uint h = 2166136261; foreach (var c in $"{x.Task}|{string.Join(' ', x.Prompt)}={x.Target}") { h ^= c; h *= 16777619; } return (int)(h % 100); }
var train = all.Where(x => Bucket(x) >= 20).ToList();
var held = all.Where(x => Bucket(x) < 20).ToList();

// relational: TRAIN the forward fact, HOLD OUT the reverse/symmetric form (inferable from the trained direction)
foreach (var (a, b) in antonyms)
{
    train.Add(new Inst("antonym", new[] { "opposite", "of", a, "=" }, b));   // forward
    held.Add(new Inst("antonym", new[] { "opposite", "of", b, "=" }, a));    // reverse — inferable: "opposite" is symmetric
}
foreach (var (c, city) in capitals)
{
    train.Add(new Inst("capital", new[] { "capital", "of", c, "=" }, city));               // forward
    held.Add(new Inst("capital", new[] { city, "is", "the", "capital", "of", "=" }, c));   // reverse lookup — inferable
}

// ---- shared vocab (built from all instances; PrismFormer seeds embeddings from the phasor codec) ----
var id = new Dictionary<string, int>(StringComparer.Ordinal) { ["<pad>"] = 0 };
var words = new List<string> { "<pad>" };
int Id(string w) { if (id.TryGetValue(w, out var i)) return i; i = id.Count; id[w] = i; words.Add(w); return i; }
foreach (var ins in train.Concat(held)) { foreach (var t in ins.Prompt) Id(t); Id(ins.Target); }
var vocab = Math.Max(1024, id.Count + 8);

(int[] Ctx, int Target) Tok(Inst x) => (x.Prompt.Select(Id).ToArray(), Id(x.Target));
var trainPairs = train.Select(Tok).ToList();

// ---- models: PrismFormer (phasor) vs a param-matched dense transformer ----
// Sized SMALL for the task on purpose: 1.1k train examples don't justify ~1M params. The efficiency thesis
// (algebra beats a dense NN pound-for-pound) is a small-regime claim — over-parameterise and both just overfit.
double[] Seed(int w) => w < words.Count ? PhasorCodec.Encode(words[w]) : new double[PhasorCodec.Dim];
var alg = new AlgFormer(vocab, shifts: 8, layers: 2, maxContext: 16,
    dModel: PhasorCodec.Dim, frozenPrefix: PhasorCodec.FrozenReals, embedSeed: Seed, seed: 1);
var targetParams = alg.ParamCount;
(int d, int L, int ff) best = (32, 2, 64); var bestDelta = long.MaxValue;
foreach (var d in new[] { 24, 32, 40, 48, 56, 64, 96, 128, 160, 192, 256 })
    foreach (var L in new[] { 2, 3, 4, 5, 6, 8 })
        foreach (var ff in new[] { 32, 48, 64, 96, 128, 160, 192, 256, 384, 512, 768 })
        {
            var probe = new MiniTransformer(vocab, dModel: d, dff: ff, layers: L, maxT: 16, seed: 42);
            var delta = Math.Abs(probe.ParamCount - targetParams);
            if (delta < bestDelta) { bestDelta = delta; best = (d, L, ff); }
        }
var xf = new MiniTransformer(vocab, dModel: best.d, dff: best.ff, layers: best.L, maxT: 16, seed: 42);

Console.WriteLine($"PrismFormer vs pound-for-pound transformer   {DateTime.Now:yyyy-MM-dd HH:mm}");
if (!AlgFormer.GradCheck(out var ar) || !MiniTransformer.GradCheck(out var xr)) { Console.WriteLine("GRADCHECK FAILED — aborting"); return; }
Console.WriteLine($"gradchecks: prismformer {ar:E1}, transformer {xr:E1} (pass)");
Console.WriteLine($"params: transformer {xf.ParamCount:N0} (d={best.d}, dff={best.ff}, L={best.L})   prismformer {alg.ParamCount:N0} (d={PhasorCodec.Dim}, S={alg.Shifts}, L={alg.Layers})   ({(double)xf.ParamCount / alg.ParamCount:F2}x pound-for-pound)");
Console.WriteLine($"tasks: {string.Join(' ', tasks)}   |   train {train.Count:N0}   held-out {held.Count:N0}   vocab {id.Count}   epochs {epochs}\n");

double Acc(Func<int[], int> predict, List<Inst> set, string task)
{
    var s = set.Where(x => x.Task == task).ToList();
    if (s.Count == 0) return double.NaN;
    var ok = 0; foreach (var x in s) { var (c, t) = Tok(x); if (predict(c) == t) ok++; }
    return ok / (double)s.Count;
}
double ComputeHeld(Func<int[], int> predict) => computeTasks.Where(t => held.Any(x => x.Task == t)).Average(t => Acc(predict, held, t));

// ---- train both identically ----
for (var ep = 1; ep <= epochs; ep++)
{
    var lr = 2e-3 * (1.0 - 0.9 * (ep - 1) / Math.Max(1, epochs - 1));
    var order = Enumerable.Range(0, trainPairs.Count).ToArray();
    var er = new Random(1000 + ep);
    for (var i = order.Length - 1; i > 0; i--) { var j = er.Next(i + 1); (order[i], order[j]) = (order[j], order[i]); }
    // the two models are independent — train them side by side on separate threads
    System.Threading.Tasks.Parallel.Invoke(
        () => { foreach (var idx in order) { var (c, t) = trainPairs[idx]; alg.TrainStep(c, t, lr); } },
        () => { foreach (var idx in order) { var (c, t) = trainPairs[idx]; xf.TrainStep(c, t, lr); } });
    if (ep % 15 == 0 || ep == epochs) Console.WriteLine($"  epoch {ep,3}/{epochs}  held-out compute: xf {ComputeHeld(xf.Predict):P0}   prism {ComputeHeld(alg.Predict):P0}");
}

// ---- report ----
Console.WriteLine("\nper-task accuracy - train / HELD-OUT --------------------------------------------");
Console.WriteLine($"  {"task",-8} {"kind",-6} {"transformer (train/held)",-28} prismformer (train/held)");
void Row(string task, string kind) { string Cell(Func<int[], int> p) => $"{Acc(p, train, task),6:P0} / {Acc(p, held, task),6:P0}"; Console.WriteLine($"  {task,-8} {kind,-6} {Cell(xf.Predict),-28} {Cell(alg.Predict)}"); }
foreach (var t in computeTasks) Row(t, "comp");
Row("copy", "algo");
foreach (var t in relationalTasks) Row(t, "rel");

Console.WriteLine("\nsummary ------------------------------------------------------------------------");
var xfC = ComputeHeld(xf.Predict); var algC = ComputeHeld(alg.Predict);
double RelTrain(Func<int[], int> p) => relationalTasks.Average(t => Acc(p, train, t));
double RelHeld(Func<int[], int> p) => relationalTasks.Average(t => Acc(p, held, t));
Console.WriteLine($"  COMPUTE     held-out (generalisation)      : transformer {xfC:P1}   prismformer {algC:P1}   ({(algC - xfC) * 100:+0.0;-0.0} pts)");
Console.WriteLine($"  copy        held-out (algorithmic)         : transformer {Acc(xf.Predict, held, "copy"):P1}   prismformer {Acc(alg.Predict, held, "copy"):P1}");
Console.WriteLine($"  relational  train    (memorised forward)   : transformer {RelTrain(xf.Predict):P1}   prismformer {RelTrain(alg.Predict):P1}");
Console.WriteLine($"  relational  held-out (inferable reverse)   : transformer {RelHeld(xf.Predict):P1}   prismformer {RelHeld(alg.Predict):P1}");
Console.WriteLine($"\n  => on UNSEEN operands PrismFormer generalises {(algC >= xfC ? "BETTER" : "worse")} on compute by {(algC - xfC) * 100:+0.0;-0.0} points, pound-for-pound.");

internal readonly record struct Inst(string Task, string[] Prompt, string Target);
