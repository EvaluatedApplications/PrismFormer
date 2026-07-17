using PrismGym;

// ═══════════════════════════════════════════════════════════════════════════════════════════════════
//  PRISM GYM — the v3 host. A gym + REPL built entirely on the Prism library (AlgFormer, the algebraic
//  transformer). No GRU, no substrate. Generates skill data, trains the Former, grades by generating.
//    prismgym train [cycles]   — run the gym, save to prism-gym.bin
//    prismgym repl             — load the model and chat (it generates continuations)
//  Default (no args): train a short run, then drop into the REPL.
// ═══════════════════════════════════════════════════════════════════════════════════════════════════

// Checkpoints under %LOCALAPPDATA%\Prism (like GenesisNova used its AppData folder), so they persist across rebuilds.
var saveDir = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Prism");
Directory.CreateDirectory(saveDir);
var save = Path.Combine(saveDir, "prism-gym.bin");
var mode = args.Length > 0 ? args[0] : "run";

// HEADLESS SWARM NODE: join a running swarm by room code and co-train + bleed with no window (see HeadlessNode / SWARM.md).
//   prismgym headless <roomCode> [dataDir]
if (mode == "headless")
    return HeadlessNode.Run(args.Length > 1 ? args[1] : "", args.Length > 2 ? args[2] : null);

// Headless HOST: open a relay room, print the code, train and fan batches out to workers that join.
//   prismgym host [dataDir]
if (mode == "host")
    return HeadlessNode.Host(args.Length > 1 ? args[1] : null);

// ANCHOR: always-on PASSIVE mesh member (bleeds + answers, no curriculum training) — for a cheap always-on box by the broker.
//   prismgym anchor <roomCode|roomName>
if (mode == "anchor")
    return HeadlessNode.Anchor(args.Length > 1 ? args[1] : "");

// BENCH: time a local Serve on THIS machine (measures how long the model takes to answer)
if (mode == "bench")
{
    var m = new StudioModel(System.IO.Path.Combine(System.IO.Path.GetTempPath(), "prismbench"));
    Console.WriteLine($"model {m.ParamCount:N0} params — timing Serve(160):");
    for (var i = 0; i < 3; i++)
    {
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var r = m.Serve("what comes after nine ");
        Console.WriteLine($"  Serve #{i + 1}: {sw.ElapsedMilliseconds} ms, {r.Continuation.Length} chars");
    }
    return 0;
}

// One-shot: chat to the running swarm from the command line — join, ask the nearest peers, print the best answer, exit.
//   prismgym ask <roomCode> <prompt…>
if (mode == "ask")
    return HeadlessNode.Ask(args.Length > 1 ? args[1] : "", args.Length > 2 ? string.Join(' ', args[2..]) : "");

// PROBE: load a SPECIFIC .bin (e.g. the anchor's model pulled down from the box) and serve prompts, to see what it
// actually produces — does an average-only anchor still emit a structured scratchpad on long arithmetic, even if wrong?
//   prismgym probe <binPath> [prompt…]     (default prompts = a few long-add questions; default path = prism-anchor.bin)
if (mode == "probe")
{
    var path = args.Length > 1 ? args[1] : Path.Combine(saveDir, "prism-anchor.bin");
    var pm = new StudioModel(Path.Combine(Path.GetTempPath(), "prismprobe"));
    if (!pm.Load(path)) { Console.WriteLine($"could not load {path}"); return 1; }
    Console.WriteLine($"loaded {path} — {pm.ParamCount:N0} params\n");
    var prompts = args.Length > 2
        ? new[] { string.Join(' ', args[2..]) }
        : new[] { "what is 47 + 38", "add 156 and 279", "23 + 19 + 44 =", "what is 128 + 384", "134 + 267 =" };
    foreach (var q in prompts)
    {
        var r = pm.Serve("user: " + q);
        Console.WriteLine($"── user: {q}");
        Console.WriteLine($"prism: {r.Continuation}");
        Console.WriteLine($"   (confidence {r.Conf:F2})\n");
    }
    return 0;
}

var cycles = args.Length > 1 && int.TryParse(args[1], out var c) ? c : (mode == "run" ? 12 : 30);

var gym = new Gym(vocab: 4096, seed: 1);
if (gym.Load(save)) Console.WriteLine($"loaded {save}");
Console.WriteLine($"AlgFormer mini-LLM: {gym.ParamCount:N0} params\n");

if (mode is "train" or "run")
{
    var sw = System.Diagnostics.Stopwatch.StartNew();
    for (var cy = 1; cy <= cycles; cy++)
    {
        var report = gym.Cycle();
        var line = string.Join("  ", report.OrderBy(kv => kv.Key).Select(kv => $"{kv.Key} L{gym.Level(kv.Key)} {kv.Value:P0}"));
        Console.WriteLine($"cycle {cy,3}/{cycles}  [{line}]  ({sw.Elapsed.TotalSeconds:F0}s, vocab {gym.VocabUsed})");
    }
    gym.Save(save);
    Console.WriteLine($"\nsaved {save}");
    // sample the model writing
    Console.WriteLine("\nsamples:");
    foreach (var q in new[] { "3 + 4 =", "12 - 5 =", "6 * 7 =", "7 in words is", "repeat the first of red blue dog :" })
        Console.WriteLine($"  {q}  ->  {gym.Complete(q, 3)}");
}

if (mode is "repl" or "run")
{
    Console.WriteLine("\nREPL — type a prompt (blank to quit). The mini-LLM writes a continuation:");
    while (true)
    {
        Console.Write("> ");
        var input = Console.ReadLine();
        if (string.IsNullOrWhiteSpace(input)) break;
        Console.WriteLine($"  {gym.Complete(input.Trim(), 10)}");
    }
}

return 0;
