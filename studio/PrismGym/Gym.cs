using PrismFormer;

namespace PrismGym;

/// <summary>
/// The Prism-native gym: it trains an <see cref="AlgFormer"/> (the algebraic transformer / mini-LLM) on skill data
/// it generates itself. Each cycle: generate a batch per active skill → tokenise (ids seeded from the codec face) →
/// expand to next-token pairs → train (data-parallel) → grade held-out probes by GENERATING → level up a skill that
/// masters its bar. No GRU, no substrate — the Former is the whole model, and every token carries its codec face.
/// </summary>
public sealed class Gym
{
    private readonly AlgFormer _model;
    private IBatchTrainer _trainer = null!;   // local cores (PrismTrainer) OR networked (MqttRelayHost) when hosting
    private MqttRelayHost? _relayHost;
    private readonly IReadOnlyList<ISkill> _skills;
    private readonly Dictionary<string, int> _id = new(StringComparer.Ordinal) { ["<unk>"] = 0 };
    private readonly List<string> _words = new() { "<unk>" };
    private readonly Dictionary<string, int> _level = new();
    private readonly Dictionary<string, Queue<double>> _window = new();
    private readonly Random _rng;
    private readonly int _ctx;
    private const double Bar = 0.85; private const int Win = 4;
    private const string Eos = "<end>";   // end-of-answer marker, trained after EVERY completion (a consistent stop signal)

    public Gym(int vocab = 4096, int seed = 1, IReadOnlyList<ISkill>? skills = null)
    {
        _model = AlgFormer.Mini(vocab, seed: seed);   // phasor face: numbers are discriminable → arithmetic decodes
        _trainer = new PrismTrainer(_model);
        _ctx = _model.Context;
        _skills = skills ?? SkillSet.Default();
        _rng = new Random(seed);
        foreach (var s in _skills) { _level[s.Name] = 1; _window[s.Name] = new Queue<double>(); }
    }

    public long ParamCount => _model.ParamCount;
    public int VocabUsed => _words.Count;
    public int Level(string skill) => _level.TryGetValue(skill, out var l) ? l : 1;

    // ---- open gym training to networked slaves (the host role) ----
    public bool Hosting => _relayHost?.Hosting ?? false;
    public int ActiveWorkers => _relayHost?.ActiveWorkers ?? 0;
    public string EnableHosting(int port, Action<string>? onEvent = null) { _relayHost = new MqttRelayHost(_model); var code = _relayHost.StartRelay(onEvent); _trainer = _relayHost; return code; }   // free public relay — no port-forward
    public void DisableHosting() { _relayHost?.StopRelay(); _relayHost = null; _trainer = new PrismTrainer(_model); }

    private int Id(string w)
    {
        if (_id.TryGetValue(w, out var i)) return i;
        if (_id.Count >= _model.Vocab) return 0;
        var id = _id.Count; _id[w] = id; _words.Add(w);
        _model.Seed(id, PhasorCodec.Encode(w));           // phasor face: identity comps frozen, orbital init
        return id;
    }
    private int[] Toks(string text) => text.Split(' ', StringSplitOptions.RemoveEmptyEntries).Select(Id).ToArray();

    /// <summary>One gym cycle: generate → train → grade → level. Returns per-skill held-out accuracy.</summary>
    public Dictionary<string, double> Cycle(int examplesPerSkill = 64, int epochs = 2, int probesPerSkill = 32, double lr = 2e-3)
    {
        // generate + expand to next-token pairs (train the whole prompt→completion sequence)
        var pairs = new List<(int[] Ctx, int Target)>();
        foreach (var s in _skills)
        {
            var lvl = _level[s.Name];
            for (var n = 0; n < examplesPerSkill; n++)
            {
                var ex = s.Train(_rng, lvl);
                var seq = Toks(ex.Prompt + " " + ex.Completion + " " + Eos);   // train it to STOP after the answer
                for (var i = 1; i < seq.Length; i++)
                    pairs.Add((seq[Math.Max(0, i - _ctx)..i], seq[i]));
            }
        }
        for (var e = 0; e < epochs; e++)
        {
            for (var i = pairs.Count - 1; i > 0; i--) { var j = _rng.Next(i + 1); (pairs[i], pairs[j]) = (pairs[j], pairs[i]); }   // shuffle
            for (var s = 0; s < pairs.Count; s += 64) _trainer.TrainBatch(pairs.GetRange(s, Math.Min(64, pairs.Count - s)), lr);   // local, or delegated to slaves when hosting
        }

        // grade held-out probes by GENERATING, and level up on mastery
        var report = new Dictionary<string, double>();
        foreach (var s in _skills)
        {
            var lvl = _level[s.Name];
            int ok = 0;
            for (var n = 0; n < probesPerSkill; n++) if (Correct(s.Probe(_rng, lvl))) ok++;
            var acc = ok / (double)probesPerSkill;
            report[s.Name] = acc;
            var w = _window[s.Name]; w.Enqueue(acc); if (w.Count > Win) w.Dequeue();
            if (w.Count >= Win && w.Average() >= Bar) { _level[s.Name] = lvl + 1; w.Clear(); }  // mastered → level up
        }
        return report;
    }

    private bool Correct(Probe p)
    {
        var ctx = Toks(p.Prompt);
        if (ctx.Length == 0) return false;
        // STRICT: the model must produce the whole answer span AND then STOP (emit <end>). We generate one token past
        // the answer and require it to be the stop marker — a model that would ramble on instead of stopping FAILS, so
        // it can't level up while producing gibberish. (Also scores multi-token answers like "twenty one" / cloze.)
        var eos = Id(Eos);
        var need = p.Answers.Max(a => a.Split(' ', StringSplitOptions.RemoveEmptyEntries).Length);
        var gen = _model.Generate(ctx, need + 1);
        var words = gen.Select(t => t > 0 && t < _words.Count ? _words[t] : "").ToArray();
        foreach (var a in p.Answers)
        {
            var at = a.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            if (at.Length < gen.Length && at.SequenceEqual(words.Take(at.Length)) && gen[at.Length] == eos) return true;
        }
        return false;
    }

    /// <summary>The mini-LLM writing: generate a continuation of the prompt (greedy), decoded to words. Input is taken
    /// as-is (no normalisation); generation stops at the end-of-answer marker if the model emits it.</summary>
    public string Complete(string prompt, int maxTokens = 8)
    {
        var ctx = Toks(prompt);
        if (ctx.Length == 0) return string.Empty;
        var eos = Id(Eos);
        var gen = _model.Generate(ctx, maxTokens);
        var words = new List<string>();
        foreach (var t in gen)
        {
            if (t == eos) break;                                   // it decided the answer is done
            words.Add(t > 0 && t < _words.Count ? _words[t] : "<unk>");
        }
        return string.Join(' ', words);
    }

    // ---- persistence ----
    public void Save(string path)
    {
        using var w = new BinaryWriter(File.Create(path));
        w.Write(0x50474D31); // "PGM1"
        w.Write(_words.Count); foreach (var word in _words) w.Write(word);
        w.Write(_skills.Count); foreach (var s in _skills) { w.Write(s.Name); w.Write(_level[s.Name]); }
        _model.Save(w);
    }
    public bool Load(string path)
    {
        if (!File.Exists(path)) return false;
        try
        {
            using var r = new BinaryReader(File.OpenRead(path));
            if (r.ReadInt32() != 0x50474D31) return false;
            var n = r.ReadInt32(); _id.Clear(); _words.Clear();
            for (var i = 0; i < n; i++) { var word = r.ReadString(); _id[word] = i; _words.Add(word); }
            var sk = r.ReadInt32(); for (var i = 0; i < sk; i++) { var name = r.ReadString(); var lvl = r.ReadInt32(); if (_level.ContainsKey(name)) _level[name] = lvl; }
            return _model.Load(r);
        }
        catch { return false; }
    }
}
