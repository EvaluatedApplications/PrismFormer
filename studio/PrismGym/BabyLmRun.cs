using System.Text;
using System.Text.Json;
using PrismFormer;

namespace PrismGym;

/// <summary>
/// A BabyLM-style challenge for the Prism <see cref="AlgFormer"/>: train a CHARACTER-level language model on a
/// budgeted, developmentally-plausible corpus (TinyStories), then score LINGUISTIC STRUCTURE — not facts — on the
/// real BLiMP grammar suite (thousands of minimal pairs; a pair is "correct" when the model gives the grammatical
/// sentence higher probability than the broken one). Designed for a long, resumable run: <see cref="Prepare"/> once,
/// then call <see cref="RunChunk"/> in a loop (cancellable, checkpointable) and <see cref="EvalBlimp"/> periodically.
/// Data is downloaded once and cached under %LOCALAPPDATA%\PrismFormer\babylm (shared with the bench harness).
/// </summary>
public sealed class BabyLmRun : IJobSource
{
    // ---- config (set before Prepare) ----
    public int WordBudget = 10_000_000;     // BabyLM Strict-Small budget: WORDS of training text (rule: <= 10,000,000)
    public int MaxEpochs = 10;              // epochs per training run (the studio trains this many MORE each Start; per-round, not cumulative)
    public int Ctx = 256;                   // characters of context (~50 words) — long enough for multi-sentence structure
    public int BlimpPerParadigm = 40;       // minimal pairs sampled per grammar paradigm (all 67 covered)
    public int Layers = 8;                  // depth = capacity (depth-scaled LR keeps deep stacks stable, see Lr())
    public int Shifts = 64;                 // relation-bank rank (S*d params/map) over the 512-dim face — base-model width
    public int BatchSize = 256;             // training minibatch — bigger = better core utilization + less GC overhead (speed knob, per-run)
    public int Seed = 1;                     // corpus-shuffle seed — SHARED via the manifest so host + workers build byte-identical corpora

    // ---- state ----
    private AlgFormer _model = null!;
    private IJobTrainer _jobTrainer = null!;   // LocalJobTrainer (local cores) OR MqttJobHost (position relay) when hosting
    private MqttJobHost? _jobHost;
    private int[] _corpus = Array.Empty<int>();
    private int[] _trainPos = Array.Empty<int>();
    private int[] _heldPos = Array.Empty<int>();
    private int _cursor;
    private char[] _chars = Array.Empty<char>();
    private Dictionary<char, int> _cid = new();
    private int _spaceId;
    private int _vocab;
    private readonly List<Para> _blimp = new();
    private readonly Random _rng = new(1);

    public int Epoch { get; private set; }
    public long Step { get; private set; }
    public bool Prepared { get; private set; }
    public long ParamCount => _model?.ParamCount ?? 0;
    public int VocabChars => _chars.Length;
    public int BlimpParadigms => _blimp.Count;
    public int BlimpPairs => _blimp.Sum(p => p.Pairs.Count);
    public double LastNextChar { get; private set; }
    public double LastBlimp { get; private set; }
    public string CorpusSource { get; private set; } = "";
    public int CorpusLen => _corpus.Length;
    public int WindowsPerEpoch => _trainPos.Length;

    // ---- IJobSource: deterministic, index-addressable data so networked workers reconstruct it locally (no data sent) ----
    public long Count => _corpus.Length;
    public (int[] Ctx, int Target) GetExample(long index)
    {
        var pos = Math.Clamp((int)index, Ctx, Math.Max(Ctx, _corpus.Length - 1));
        var c = new int[Ctx]; Array.Copy(_corpus, pos - Ctx, c, 0, Ctx); return (c, _corpus[pos]);
    }
    public byte[] Manifest() { using var ms = new MemoryStream(); var w = new BinaryWriter(ms); w.Write((byte)1); w.Write(WordBudget); w.Write(Ctx); w.Write(Seed); w.Flush(); return ms.ToArray(); }

    /// <summary>Build a corpus-only source from a BabyLM manifest — for a networked worker to reconstruct the identical
    /// data locally (downloads the same file once, seeded shuffle -> byte-identical corpus).</summary>
    public static BabyLmRun FromManifest(byte[] m, Action<string>? log = null)
    {
        using var r = new BinaryReader(new MemoryStream(m)); r.ReadByte();
        var run = new BabyLmRun { WordBudget = r.ReadInt32(), Ctx = r.ReadInt32(), Seed = r.ReadInt32() };
        run.PrepareCorpusOnly(log ?? (_ => { }));
        return run;
    }

    /// <summary>Load the budgeted BabyLM corpus text (downloads/reads the Parquet, word-capped, normalised) WITHOUT
    /// building a model — for exporting starter corpus files (see StarterData).</summary>
    public string CorpusText(Action<string> log)
    {
        using var http = new HttpClient { Timeout = TimeSpan.FromSeconds(120) };
        http.DefaultRequestHeaders.UserAgent.ParseAdd("PrismFormer-BabyLM/1.0");
        return LoadCorpus(http, log);
    }

    public void PrepareCorpusOnly(Action<string> log)
    {
        using var http = new HttpClient { Timeout = TimeSpan.FromSeconds(120) };
        http.DefaultRequestHeaders.UserAgent.ParseAdd("PrismFormer-BabyLM/1.0");
        var text = LoadCorpus(http, log);
        _chars = Enumerable.Range(32, 95).Select(i => (char)i).ToArray(); var V = _chars.Length;
        _cid = new Dictionary<char, int>(); for (var i = 0; i < V; i++) _cid[_chars[i]] = i;
        _spaceId = _cid.TryGetValue(' ', out var sp) ? sp : 0; _vocab = Math.Max(64, V + 4);
        _corpus = new int[text.Length]; for (var i = 0; i < text.Length; i++) _corpus[i] = Cid(text[i]);
        Prepared = true;
        log($"corpus ready: {_corpus.Length:N0} chars ({CorpusSource})");
    }

    private sealed class Para { public string Name = ""; public string Cat = ""; public List<(string g, string b)> Pairs = new(); }

    private static string Cache
    {
        get { var d = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "PrismFormer", "babylm"); Directory.CreateDirectory(d); return d; }
    }

    /// <summary>Fold text to a small consistent ascii-ish character set so corpus and BLiMP share one vocabulary.</summary>
    public static string Norm(string s)
    {
        var sb = new StringBuilder(s.Length);
        foreach (var ch in s)
        {
            var c = ch switch
            {
                '‘' or '’' or '′' => '\'',
                '“' or '”' => '"',
                '–' or '—' or '−' => '-',
                '\t' or '\r' or '\n' => ' ',
                _ => ch
            };
            if (c < 32 || c > 126) c = ' ';
            sb.Append(c);
        }
        var raw = sb.ToString(); var outp = new StringBuilder(raw.Length); var prev = false;
        foreach (var c in raw) { var sp = c == ' '; if (sp && prev) continue; outp.Append(c); prev = sp; }
        return outp.ToString();
    }

    private int Cid(char ch) => _cid.TryGetValue(ch, out var i) ? i : _spaceId;
    private void Shuffle(int[] a) { for (var i = a.Length - 1; i > 0; i--) { var j = _rng.Next(i + 1); (a[i], a[j]) = (a[j], a[i]); } }

    // ════════════════ preparation (download + build), run on a worker thread ════════════════
    public void Prepare(Action<string> log, CancellationToken ct)
    {
        using var http = new HttpClient { Timeout = TimeSpan.FromSeconds(120) };
        http.DefaultRequestHeaders.UserAgent.ParseAdd("PrismFormer-BabyLM/1.0");

        var text = LoadCorpus(http, log);   // already word-budgeted to WordBudget words
        ct.ThrowIfCancellationRequested();
        LoadBlimp(http, log);
        ct.ThrowIfCancellationRequested();

        // FIXED printable-ASCII vocabulary: Norm() folds ALL text into bytes 32..126 (95 chars), so the vocab is
        // identical for every corpus and every word budget. This keeps checkpoints compatible — Load always matches —
        // instead of depending on which characters a particular random sample happened to include (the old brittle bug).
        _chars = Enumerable.Range(32, 95).Select(i => (char)i).ToArray(); var V = _chars.Length;
        _cid = new Dictionary<char, int>(); for (var i = 0; i < V; i++) _cid[_chars[i]] = i;
        _spaceId = _cid.TryGetValue(' ', out var sp) ? sp : 0;
        _vocab = Math.Max(64, V + 4);

        _corpus = new int[text.Length]; for (var i = 0; i < text.Length; i++) _corpus[i] = Cid(text[i]);
        var split = Math.Max(Ctx + 1, (int)(_corpus.Length * 0.95));
        _trainPos = Enumerable.Range(Ctx, Math.Max(0, split - Ctx)).ToArray();
        _heldPos = Enumerable.Range(split, Math.Max(0, _corpus.Length - split)).ToArray();
        Shuffle(_trainPos); _cursor = 0; Epoch = 0; Step = 0;

        double[] Seed(int w) => w < V ? PhasorCodec.Encode(_chars[w].ToString()) : new double[PhasorCodec.Dim];
        _model = new AlgFormer(_vocab, shifts: Shifts, layers: Layers, maxContext: Ctx, dModel: PhasorCodec.Dim,
            frozenPrefix: PhasorCodec.FrozenReals, embedSeed: Seed, seed: 1);
        _jobTrainer = new LocalJobTrainer(_model, this);   // local cores (PrismTrainer/EvalApp inside); swapped to MqttJobHost when hosting

        Prepared = true;
        log($"ready: corpus {_corpus.Length:N0} chars ({CorpusSource}), vocab {V} chars, BLiMP {_blimp.Count} paradigms / {BlimpPairs} pairs, params {ParamCount:N0}, batch {BatchSize}, seed {Seed}");
    }

    // The official BabyLM 2025 Strict-Small corpus (10 developmentally-plausible domains) ships as one Parquet file.
    private const string BabyLmParquetUrl =
        "https://huggingface.co/datasets/PatrickHaller/BabyLM2025-Strict-Small-Dataset/resolve/main/data/train-00000-of-00001.parquet";

    private string LoadCorpus(HttpClient http, Action<string> log)
    {
        var f = Path.Combine(Cache, "babylm-strict-small.parquet");
        if (!File.Exists(f))
        {
            try
            {
                log("downloading official BabyLM Strict-Small corpus (Parquet) ...");
                var bytes = http.GetByteArrayAsync(BabyLmParquetUrl).GetAwaiter().GetResult();
                File.WriteAllBytes(f, bytes);
                log($"  cached {bytes.Length / (1024 * 1024)} MB");
            }
            catch (Exception e) { log($"  BabyLM corpus download failed ({e.Message.Split('\n')[0]})"); }
        }
        if (File.Exists(f))
        {
            try
            {
                var text = ReadParquetWords(f, WordBudget, Seed, log);
                if (text.Length > 0) { CorpusSource = $"BabyLM Strict-Small ({WordBudget:N0} words)"; return Norm(text); }
            }
            catch (Exception e) { log($"  Parquet read failed ({e.Message.Split('\n')[0]})"); }
        }

        // fallbacks — NON-official, clearly flagged so no run is mistaken for a compliant BabyLM entry
        var wiki = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "GenesisNova", "datasets", "wikimedia_wikipedia_20231101.en_train_text.txt");
        if (File.Exists(wiki)) { CorpusSource = "wiki-cache (NON-official)"; return Norm(WordCap(File.ReadAllText(wiki, Encoding.UTF8), WordBudget)); }

        CorpusSource = "embedded (NON-official)";
        var seed = "The little girl saw a big dog in the park. She was happy because the sun was warm and the birds were singing. " +
                   "Her mother said that they would go home soon. The boy ran to the tree and climbed up high to see the river. ";
        var sb = new StringBuilder(); while (sb.Length < 200_000) sb.Append(seed); return Norm(sb.ToString());
    }

    private sealed class BabyLmRow { public string? text { get; set; } public string? source { get; set; } }

    /// <summary>Read the "text" column of the BabyLM Parquet (via ParquetSerializer), concatenating documents until
    /// <paramref name="wordBudget"/> whitespace-delimited words are collected (the budget is measured in WORDS). The
    /// documents are SHUFFLED first (fixed seed, so it's reproducible and resume-safe) so a budget smaller than the full
    /// corpus draws a representative sample spread across all 10 domains, not just the first ones in file order.</summary>
    private static string ReadParquetWords(string path, int wordBudget, int seed, Action<string> log)
    {
        using var stream = File.OpenRead(path);
        var rows = Parquet.Serialization.ParquetSerializer.DeserializeAsync<BabyLmRow>(stream).GetAwaiter().GetResult().Data;
        log($"  {rows.Count:N0} documents in corpus");
        var order = Enumerable.Range(0, rows.Count).ToArray();
        var rng = new Random(seed);   // SEEDED shuffle: host + workers build BYTE-IDENTICAL corpora from the shared manifest seed
                                  // every time, so repeated small runs stream diverse data across the whole corpus over
                                  // rounds (vocab is fixed to ASCII, so a changing sample never breaks checkpoint reload)
        for (var i = order.Length - 1; i > 0; i--) { var j = rng.Next(i + 1); (order[i], order[j]) = (order[j], order[i]); }

        var sb = new StringBuilder();
        long words = 0;
        foreach (var k in order)
        {
            if (words >= wordBudget) break;
            var t = rows[k].text;
            if (string.IsNullOrEmpty(t)) continue;
            foreach (var w in t.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries))
            {
                if (words >= wordBudget) break;
                sb.Append(w); sb.Append(' '); words++;
            }
            sb.Append('\n');
        }
        log($"  {words:N0} words loaded ({(words < wordBudget ? "full corpus" : "random sample across domains")})");
        return sb.ToString();
    }

    private static string WordCap(string text, int wordBudget)
    {
        var sb = new StringBuilder(); long words = 0;
        foreach (var w in text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries))
        {
            if (words >= wordBudget) break;
            sb.Append(w); sb.Append(' '); words++;
        }
        return sb.ToString();
    }

    private void LoadBlimp(HttpClient http, Action<string> log)
    {
        _blimp.Clear();
        var dir = Path.Combine(Cache, "blimp"); Directory.CreateDirectory(dir);
        var files = Directory.GetFiles(dir, "*.jsonl").ToList();
        if (files.Count == 0)
        {
            try
            {
                log("downloading BLiMP grammar paradigms ...");
                var list = http.GetStringAsync("https://api.github.com/repos/alexwarstadt/blimp/contents/data").GetAwaiter().GetResult();
                using var doc = JsonDocument.Parse(list);
                foreach (var el in doc.RootElement.EnumerateArray())
                {
                    var name = el.GetProperty("name").GetString() ?? "";
                    if (!name.EndsWith(".jsonl")) continue;
                    var url = el.GetProperty("download_url").GetString(); if (url == null) continue;
                    File.WriteAllText(Path.Combine(dir, name), http.GetStringAsync(url).GetAwaiter().GetResult());
                }
                files = Directory.GetFiles(dir, "*.jsonl").ToList();
            }
            catch (Exception e) { log($"  BLiMP download failed ({e.Message.Split('\n')[0]}) - using embedded mini-suite"); }
        }

        foreach (var path in files.OrderBy(x => x))
        {
            var p = new Para { Name = Path.GetFileNameWithoutExtension(path) };
            foreach (var line in File.ReadLines(path))
            {
                if (p.Pairs.Count >= BlimpPerParadigm) break;
                if (string.IsNullOrWhiteSpace(line)) continue;
                try
                {
                    using var d = JsonDocument.Parse(line); var r = d.RootElement;
                    var g = Norm(r.GetProperty("sentence_good").GetString() ?? "");
                    var b = Norm(r.GetProperty("sentence_bad").GetString() ?? "");
                    if (p.Cat == "" && r.TryGetProperty("linguistics_term", out var lt)) p.Cat = lt.GetString() ?? "";
                    if (g.Length > 0 && b.Length > 0) p.Pairs.Add((g, b));
                }
                catch { }
            }
            if (p.Pairs.Count > 0) { if (p.Cat == "") p.Cat = p.Name; _blimp.Add(p); }
        }
        if (_blimp.Count > 0) return;

        (string g, string b, string cat)[] mini =
        {
            ("The children are playing outside.",       "The children is playing outside.",       "subject_verb_agr"),
            ("My sister has finished her homework.",     "My sister have finished her homework.",  "subject_verb_agr"),
            ("These books are on the table.",            "This books are on the table.",           "det_noun_agr"),
            ("He hurt himself while running.",           "He hurt herself while running.",         "anaphor_agr"),
            ("The girls saw themselves in the mirror.",  "The girls saw herself in the mirror.",   "anaphor_agr"),
            ("Nobody has ever been there.",              "Nobody has never been there.",           "npi"),
            ("The dog that chased the cat is tired.",    "The dog that chased the cat are tired.", "rel_clause_agr"),
            ("She gave the book to me.",                 "She gave the book to I.",                "case"),
            ("There is a problem with the plan.",        "There are a problem with the plan.",     "existential_agr"),
            ("I have two red apples.",                   "I have two red apple.",                  "number"),
        };
        foreach (var gr in mini.GroupBy(m => m.cat))
            _blimp.Add(new Para { Name = gr.Key, Cat = gr.Key, Pairs = gr.Select(m => (Norm(m.g), Norm(m.b))).ToList() });
    }

    // ════════════════ training ════════════════
    public double Lr()
    {
        const double warm = 5_000;
        var peak = 1.5e-3 * Math.Min(1.0, 4.0 / Layers);   // depth-scaled: deeper stacks need a gentler peak LR
        var b = Step < warm ? peak * (Step / warm) : peak;
        return Math.Max(2e-4, b * Math.Max(0.3, 1.0 - 0.05 * Epoch));
    }

    /// <summary>Train one bounded chunk of windows in small sub-batches (data-parallel), checking <paramref name="ct"/>
    /// between sub-batches so a Stop lands promptly — between batches, never at an epoch boundary.</summary>
    public void RunChunk(int chunkWindows, double lr, CancellationToken ct = default)
    {
        var done = 0;
        while (done < chunkWindows && !ct.IsCancellationRequested)
        {
            var take = Math.Min(BatchSize, chunkWindows - done);   // one minibatch, expressed as POSITIONS
            var positions = new List<long>(take);
            for (var i = 0; i < take; i++)
            {
                if (_cursor >= _trainPos.Length) { Epoch++; Shuffle(_trainPos); _cursor = 0; }
                positions.Add(_trainPos[_cursor++]);
            }
            // LOCAL cores, or delegated to networked slaves when hosting — only positions travel; workers reconstruct
            // the windows locally from the same corpus, and only the model + gradients are the payload.
            _jobTrainer.TrainPositions(positions, lr);
            Step += take;
            done += take;
        }
    }

    // ---- open this run to networked slaves over the free public relay (the host role) ----
    public bool Hosting => _jobHost?.Hosting ?? false;
    public int ActiveWorkers => _jobHost?.ActiveWorkers ?? 0;

    /// <summary>Open this training run to slaves via the free public relay (dials out to an MQTT broker — works through
    /// any NAT, no port-forward). Only the manifest + model + gradients cross the wire; workers reconstruct the data
    /// from the same corpus. Returns the pasteable room code. Call after <see cref="Prepare"/>. (<paramref name="port"/> ignored.)</summary>
    public string EnableHosting(int port, Action<string>? onEvent = null)
    {
        _jobHost = new MqttJobHost(_model, this, Manifest());
        var code = _jobHost.StartRelay(onEvent);
        _jobTrainer = _jobHost;
        return code;
    }

    public void DisableHosting() { _jobHost?.StopRelay(); _jobHost = null; _jobTrainer = new LocalJobTrainer(_model, this); }

    // ════════════════ evaluation ════════════════
    public double EvalNextChar(int sample = 1500)
    {
        if (_heldPos.Length == 0) return LastNextChar;
        int ok = 0, n = 0; var stride = Math.Max(1, _heldPos.Length / sample);
        for (var k = 0; k < _heldPos.Length; k += stride)
        {
            var p = _heldPos[k]; var c = new int[Ctx]; Array.Copy(_corpus, p - Ctx, c, 0, Ctx);
            if (_model.Predict(c) == _corpus[p]) ok++; n++;
        }
        LastNextChar = n > 0 ? ok / (double)n : 0; return LastNextChar;
    }

    private double LogProb(string s)
    {
        double total = 0; var ctx = new int[Ctx]; for (var k = 0; k < Ctx; k++) ctx[k] = _spaceId;
        foreach (var ch in s)
        {
            var tgt = Cid(ch); var lg = _model.LogitsFor(ctx);
            var mx = double.NegativeInfinity; for (var w = 0; w < lg.Length; w++) if (lg[w] > mx) mx = lg[w];
            var sum = 0.0; for (var w = 0; w < lg.Length; w++) sum += Math.Exp(lg[w] - mx);
            total += lg[tgt] - mx - Math.Log(sum);
            Array.Copy(ctx, 1, ctx, 0, Ctx - 1); ctx[Ctx - 1] = tgt;
        }
        return total;
    }

    /// <summary>Score BLiMP: macro accuracy over paradigms (parallel across the i9's cores; forward is read-only).</summary>
    public (double overall, Dictionary<string, double> byCategory) EvalBlimp(int capPerParadigm = int.MaxValue)
    {
        var per = new double[_blimp.Count];
        System.Threading.Tasks.Parallel.For(0, _blimp.Count, i =>
        {
            var p = _blimp[i]; var n = Math.Min(capPerParadigm, p.Pairs.Count); var ok = 0;
            for (var j = 0; j < n; j++) { var (g, b) = p.Pairs[j]; if (LogProb(g) > LogProb(b)) ok++; }
            per[i] = n > 0 ? ok / (double)n : 0;
        });
        var byCat = _blimp.Select((p, i) => (p.Cat, acc: per[i])).GroupBy(x => x.Cat)
            .ToDictionary(g => g.Key, g => g.Average(x => x.acc));
        LastBlimp = per.Length > 0 ? per.Average() : 0;
        return (LastBlimp, byCat);
    }

    /// <summary>Greedy character continuation, so the run can be eyeballed for coherence.</summary>
    public string Sample(string seed, int n)
    {
        var s = Norm(seed); if (s.Length == 0) s = " ";
        var ctx = new int[Ctx]; for (var k = 0; k < Ctx; k++) ctx[k] = _spaceId;
        var start = s.Length > Ctx ? s[^Ctx..] : s; var off = Ctx - start.Length;
        for (var k = 0; k < start.Length; k++) ctx[off + k] = Cid(start[k]);
        var sb = new StringBuilder(s);
        for (var i = 0; i < n; i++)
        {
            var t = _model.Predict(ctx);
            sb.Append(t >= 0 && t < _chars.Length ? _chars[t] : ' ');
            Array.Copy(ctx, 1, ctx, 0, Ctx - 1); ctx[Ctx - 1] = t;
        }
        return sb.ToString();
    }

    // ════════════════ persistence (model + vocab + training position) ════════════════
    public void Save(string path)
    {
        if (!Prepared) return;
        using var w = new BinaryWriter(File.Create(path));
        w.Write(0x42424C31); // "BBL1"
        w.Write(Ctx); w.Write(Layers); w.Write(Shifts);
        w.Write(_chars.Length); foreach (var c in _chars) w.Write(c);
        w.Write(Epoch); w.Write(Step); w.Write(_cursor);
        _model.Save(w);
    }

    public bool Load(string path)
    {
        if (!Prepared || !File.Exists(path)) return false;
        try
        {
            using var r = new BinaryReader(File.OpenRead(path));
            if (r.ReadInt32() != 0x42424C31) return false;
            if (r.ReadInt32() != Ctx || r.ReadInt32() != Layers || r.ReadInt32() != Shifts) return false;
            var nc = r.ReadInt32(); if (nc != _chars.Length) return false;
            for (var i = 0; i < nc; i++) if (r.ReadChar() != _chars[i]) return false;
            Epoch = r.ReadInt32(); Step = r.ReadInt64(); _cursor = Math.Min(r.ReadInt32(), _trainPos.Length);
            return _model.Load(r);
        }
        catch { return false; }
    }

    /// <summary>Peek a checkpoint's context length without loading it, so a resume can build the model at the SAVED ctx
    /// (word budget / pairs can change freely between reinforcement rounds; only ctx is baked into the architecture).</summary>
    public static int? PeekCtx(string path)
    {
        try
        {
            using var r = new BinaryReader(File.OpenRead(path));
            if (r.ReadInt32() != 0x42424C31) return null;
            return r.ReadInt32();   // ctx is the first field after the magic
        }
        catch { return null; }
    }
}
