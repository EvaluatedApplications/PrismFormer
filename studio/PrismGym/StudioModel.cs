using System.Collections.Concurrent;
using System.Diagnostics;
using System.Text;
using PrismFormer;
using PrismFormer.Gpu;

namespace PrismGym;

/// <summary>
/// The ONE production model behind Prism Studio (see STUDIO.md / PrismSpec). Frozen architecture; the only knobs are
/// the training-data folder and how long you train.
///
/// CONCURRENCY (see SWARM.md): single-writer + snapshot. The training loop is the ONLY thread that mutates the live
/// weights, under <c>_write</c>. Serving — the REPL and swarm queries — reads a lock-free immutable <c>_snapshot</c>
/// that the writer republishes periodically, so it never blocks training and never sees a torn model. Network-fed
/// learning (bled pairs, backprop-on-wrong) is enqueued to <c>_inbox</c> and drained by the writer, so there is
/// exactly one writer and no race. Control ops (save/load/host) serialize against the writer on the same lock.
/// </summary>
public sealed class StudioModel
{
    readonly CharVocab _v = new();
    readonly AlgFormer _model;                       // MUTABLE — mutated only under _write
    volatile AlgFormer _snapshot;                    // immutable for readers; republished by the writer
    readonly object _write = new();                  // serializes every write to _model + control ops
    volatile bool _real;                             // has meaningful weights (successfully LOADED or TRAINED) — a fresh boot after a failed load is NOT real, and must never be autosaved over a good checkpoint
    readonly ConcurrentQueue<(int[] Ctx, int Target)> _inbox = new();   // network-fed examples → the single writer
    GossipInbox _gossip = null!;
    string[]? _pairCache;
    IBatchTrainer _trainer;
    MqttRelayHost? _host;

    public string DataDir { get; }
    public string CorpusDir => Path.Combine(DataDir, "text");
    public string PairsDir => Path.Combine(DataDir, "pairs");
    public string GossipDir => Path.Combine(DataDir, "gossip");
    public string ChatDir => Path.Combine(DataDir, "chat");   // your REPL conversation → trained WITH its context, own fair share
    public string GroupDir => Path.Combine(DataDir, "group"); // the network group chat → rolling, context-capped, trained as (context → next HUMAN turn)

    /// <summary>Folder sampling weight = source.Count^MixAlpha. 0 = uniform per folder (tiny pools oversampled, the big
    /// corpus starved — the old behaviour); 1 = fully volume-proportional (corpus dominates). ~0.5 gives the corpus the
    /// share its volume warrants while a few hundred pairs still get ample ABSOLUTE repetition (few items × small share
    /// is still many reps each). Tune live and watch the "[mix]" log line for the resulting per-folder shares.</summary>
    public double MixAlpha { get; set; } = 0.5;
    /// <summary>When set, train ONLY on the text corpus (data/text): mutes pairs/chat/group/gossip/qa/peer AND the peer
    /// inbox, so language modelling isn't crowded out by the small high-signal pools. The isolation experiment.</summary>
    public bool CorpusOnly { get; set; }

    readonly object _group = new();
    string _groupBuf = "";                                    // rolling group-chat log, capped at the context window
    string GroupFile => Path.Combine(GroupDir, "group.txt");

    public StudioModel(string dataDir)
    {
        DataDir = dataDir;
        Directory.CreateDirectory(CorpusDir); Directory.CreateDirectory(PairsDir); Directory.CreateDirectory(GossipDir); Directory.CreateDirectory(ChatDir); Directory.CreateDirectory(GroupDir);
        try { if (File.Exists(GroupFile)) _groupBuf = CapRecent(File.ReadAllText(GroupFile), PrismSpec.Context); } catch { }
        _gossip = new GossipInbox(GossipDir);
        _model = PrismSpec.NewModel(Seed);
        _trainer = new PrismTrainer(_model);
        _snapshot = Clone(_model);
    }

    static string Tail(string s, int n) => s.Length > n ? s[^n..] : s;   // keep the last n chars (rolling cap)

    /// <summary>Roll a chat/group log to the recent context window: keep the last <paramref name="max"/> chars but cut at
    /// a TURN boundary (drop the leading partial line) so it reloads clean. Both the group log and the REPL transcript are
    /// capped to this — the saved context file is never larger than the model's context window.</summary>
    public static string CapRecent(string t, int max)
    {
        if (t.Length <= max) return t;
        var i = t.IndexOf('\n', t.Length - max);   // first turn boundary at/after (len-max)
        return i >= 0 && i + 1 < t.Length ? t[(i + 1)..] : t[^max..];
    }

    // canonical codec face for a token id — the FROZEN identity prefix holds its exact value; used to seed AND to restore
    double[] Seed(int w) => PhasorCodec.Encode(_v.Symbol(w));   // char OR subword text → its codec face (number face if it parses as a number, else signature); frozen prefix holds the exact identity

    static AlgFormer Clone(AlgFormer m) { var c = AlgFormer.Deserialize(m.Serialize()); c.Map = PrismEval.Cpu; return c; }   // the lock-free SERVING snapshot uses the EvalApp CPU-gated forward; the training _model stays sequential (it's already batch-parallel)
    static string Tail(string s) => (s.Length > 90 ? "…" + s[^90..] : s).TrimStart();   // log preview only — the ellipsis marks that the FULL context (up to 256) is what's trained on
    static string Clean(string s) => s.Replace('\n', ' ').Replace('\r', ' ');

    string GenAnswer(string prompt, int max) => GenerateCached(_snapshot, prompt, max);   // off the snapshot; stops at STOP

    /// <summary>Confidence-gated causal generation via the KV cache (O(T)/token). Left-aligns the prompt at natural
    /// positions and reserves window room so a typical STOP-terminated reply never leaves the fast incremental path;
    /// only an over-long reply falls back to sliding recompute. Stops at the STOP token. Per token it is GREEDY when
    /// the codec decode is sharp (arithmetic / a known answer stays EXACT) and SAMPLES only when genuinely uncertain
    /// (a real chat branch), which is what breaks greedy's repetition loops. The codec's own peakedness routes it,
    /// token by token: no task classifier, nothing keyed off a cue token.</summary>
    const double DecodeConfident = 0.60;   // top-1 probability at/above which we commit greedily (an exact decode)
    const double DecodeTemp = 0.80;        // sampling temperature used ONLY when below that confidence
    static int PickToken(double[] lg)
    {
        var top = 0; var max = lg[0];
        for (var i = 1; i < lg.Length; i++) if (lg[i] > max) { max = lg[i]; top = i; }
        double sum = 0; for (var i = 0; i < lg.Length; i++) sum += Math.Exp(lg[i] - max);
        if (1.0 / sum >= DecodeConfident) return top;                       // p(top) = 1/sum ; confident -> greedy (exact)
        var tm = double.NegativeInfinity;                                   // uncertain -> sample at temperature
        for (var i = 0; i < lg.Length; i++) { var s = lg[i] / DecodeTemp; if (s > tm) tm = s; }
        var tp = new double[lg.Length]; double tsum = 0;
        for (var i = 0; i < lg.Length; i++) { tp[i] = Math.Exp(lg[i] / DecodeTemp - tm); tsum += tp[i]; }
        var r = Random.Shared.NextDouble() * tsum; double acc = 0;
        for (var i = 0; i < lg.Length; i++) { acc += tp[i]; if (acc >= r) return i; }
        return top;
    }
    string GenerateCached(AlgFormer m, string prompt, int n)
    {
        var ids = _v.Encode(prompt);
        var reserve = Math.Min(n, PrismSpec.Context / 3);                          // leave room so generation stays incremental
        var primeLen = Math.Min(ids.Length, PrismSpec.Context - reserve);
        var window = new List<int>(primeLen > 0 ? ids[(ids.Length - primeLen)..] : new[] { CharVocab.Pad });
        var cache = m.NewCache();
        var lg = m.Prime(cache, window.ToArray());
        var sb = new StringBuilder();
        for (var k = 0; k < n; k++)
        {
            var t = PickToken(lg); if (t == CharVocab.End) break;
            sb.Append(_v.Symbol(t)); window.Add(t);   // Symbol → the token's text (a subword emits its whole 2..4 chars)
            if (cache.Length < PrismSpec.Context) lg = m.Step(cache, t);          // O(T) fast path
            else lg = m.Prime(cache, window.GetRange(window.Count - (PrismSpec.Context - 1), PrismSpec.Context - 1).ToArray());   // window full → slide-recompute
        }
        return sb.ToString();
    }
    static string[] RawLines(string dir) => Directory.Exists(dir) ? Directory.EnumerateFiles(dir).Where(f => f.EndsWith(".tsv") || f.EndsWith(".pairs")).SelectMany(File.ReadLines).Where(l => l.Contains('\t')).ToArray() : Array.Empty<string>();
    public long ParamCount => _model.ParamCount;     // structural (lengths) — safe to read lock-free
    public string Signature => PrismSpec.Signature;
    public CharVocab Vocab => _v;

    IJobSource BuildSource()
    {
        var corpus = CorpusSource.FromFolders(PrismSpec.Context, _v, CorpusDir);
        var pairs = PairSource.FromFolders(PrismSpec.Context, _v, PairsDir, GossipDir);
        return new MixSource(corpus, pairs);
    }
    public (long Corpus, long Pairs) Counts()
        => (CorpusSource.FromFolders(PrismSpec.Context, _v, CorpusDir).Count, PairSource.FromFolders(PrismSpec.Context, _v, PairsDir, GossipDir).Count);

    // ════════ THE WRITER — the only thread that mutates the live weights ════════
    public void Train(int epochs, double minutesPerEpoch, Action<string> report, Action<string> log, CancellationToken ct)
    {
        report("loading training data…");
        var corpus = CorpusSource.FromFolders(PrismSpec.Context, _v, CorpusDir);   // load the big corpus ONCE, reuse every epoch (was reloading 53 MB per epoch)
        if (corpus.Count == 0 && (CorpusOnly || (PairSource.FromFolders(PrismSpec.Context, _v, PairsDir, GossipDir).Count == 0 && _inbox.IsEmpty)))
        { report(CorpusOnly ? "corpus-only: add files to data/text" : "no training data — add files to data/text or data/pairs"); return; }
        if (CorpusOnly) log("[train] CORPUS-ONLY — training on data/text alone (pairs/chat/group/gossip/peer + inbox muted)");
        report("training…"); log("[train] started");
        // GPU acceleration: build a GpuTrainer once (kernels compile ~1-2s) if a CUDA GPU is present and we're not the
        // relay host. CPU model stays the source of truth (serve/bleed/save read it); the GPU just does fwd+bwd ~10x
        // faster. Falls back to CPU per-batch on any GPU error (e.g. a full-1024 window OOMing the card).
        GpuTrainer? gpu = null;
        try { if (GpuDevice.HasGpu) { gpu = new GpuTrainer(_model); log($"[train] GPU acceleration ON — {GpuDevice.Describe}"); } else log("[train] no CUDA GPU — training on CPU"); }
        catch (Exception e) { log("[train] GPU init failed → CPU: " + e.Message.Split('\n')[0]); gpu = null; }
        // FULL speed: no priority gimping (that was the ~60% CPU). EvalApp is data-parallel and schedules any overflow.
        // WARMUP: the first batch is TINY so feedback lands in seconds, then it doubles up to fill every core.
        var lr = 1.5e-3 * Math.Min(1.0, 4.0 / PrismSpec.Layers);
        var rng = new Random();
        var maxBatch = gpu != null ? 512 : Math.Max(64, Environment.ProcessorCount);   // GPU wants big batches (amortises the per-batch param-sync); CPU stays at ~cores
        var warm = 4;
        var epochSecs = Math.Max(5.0, minutesPerEpoch * 60.0);   // an "epoch" = ~minutesPerEpoch of random draws — the corpus is far too big for a full pass
        var sw = Stopwatch.StartNew(); long stepTotal = 0; double lastSnap = 0, lastSample = 0, loss = 0; var first = true; var sampleTick = 0; var genBusy = false;
        if (corpus.Count > 0) try { var (e0, t0) = corpus.GetExample(rng.NextInt64(corpus.Count)); log($"…{Clean(Tail(_v.Decode(e0)))} ▸ \"{_v.Symbol(_snapshot.Predict(e0))}\"  (real \"{_v.Symbol(t0)}\")"); } catch { }   // instant: untrained guess (Symbol → char or subword)
        for (var ep = 0; ep < epochs && !ct.IsCancellationRequested; ep++)
        {
            // one source PER FOLDER, then sampled by VOLUME (weight = Count^MixAlpha) so the big corpus gets the share its
            // size warrants instead of a flat 1/N that let a few dozen chat/group lines punch at the corpus's weight.
            var srcs = new List<(string Name, IJobSource S)>();
            if (corpus.Count > 0) srcs.Add(("text", corpus));
            var chatText = Directory.Exists(ChatDir) ? string.Concat(Directory.EnumerateFiles(ChatDir, "*.txt").OrderBy(f => f).Select(File.ReadAllText)) : "";
            if (!CorpusOnly)
            {
                var pairsS = PairSource.FromFolders(PrismSpec.Context, _v, PairsDir); if (pairsS.Count > 0) srcs.Add(("pairs", pairsS));
                var gossipS = PairSource.FromFolders(PrismSpec.Context, _v, GossipDir); if (gossipS.Count > 0) srcs.Add(("gossip", gossipS));
                var qa = PairSource.ReadFolders(PairsDir, GossipDir);   // the SAME curriculum Q&A, ALSO framed as the REPL/group serve it ("prism: Q\nprism: " → A) so PRISM answers questions in chat (raw lanes above still feed the gym probe / `ask`)
                if (qa.Count > 0) { var qaChatS = new PairSource(PrismSpec.Context, GroupChat.AsChat(qa), _v); if (qaChatS.Count > 0) srcs.Add(("qa-chat", qaChatS)); }
                var chatS = PairSource.FromChat(PrismSpec.Context, _v, chatText); if (chatS.Count > 0) srcs.Add(("chat", chatS));   // progressive (conversation → your reply) pairs, own fair share
                var groupS = new PairSource(PrismSpec.Context, GroupChat.Pairs(GroupContext), _v); if (groupS.Count > 0) srcs.Add(("group", groupS));   // network group chat → (context → next HUMAN turn); STOP appended by PairSource
                var distill = _peerDistill.ToArray();   // peers' personality distilled onto OUR chat contexts — trained exactly like the chat source (context → reply)
                if (distill.Length > 0) { var distillS = new PairSource(PrismSpec.Context, distill.Select(d => (d.Ctx, d.Reply)), _v); if (distillS.Count > 0) srcs.Add(("peer", distillS)); }
            }
            if (srcs.Count == 0) break;
            // VOLUME-WEIGHTED folder mix: weight ∝ Count^MixAlpha. Recomputed per epoch (pools grow via bleed); logged once.
            var mixW = srcs.Select(x => Math.Pow(Math.Max(1, x.S.Count), MixAlpha)).ToArray();
            var mixSum = mixW.Sum();
            if (ep == 0) log($"[mix] alpha={MixAlpha:0.##}  " + string.Join("  ", srcs.Select((x, i) => $"{x.Name} {mixW[i] / mixSum:P0}")));
            int PickSrc() { var r = rng.NextDouble() * mixSum; double a = 0; for (var i = 0; i < mixW.Length; i++) { a += mixW[i]; if (r <= a) return i; } return mixW.Length - 1; }
            var epStart = sw.Elapsed.TotalSeconds;
            while (!ct.IsCancellationRequested && sw.Elapsed.TotalSeconds - epStart < epochSecs)
            {
                var bsz = Math.Min(maxBatch, warm); warm = Math.Min(maxBatch, warm * 2);   // warmup ramp: 4, 8, 16, … → maxBatch
                var batch = new List<(int[] Ctx, int Target)>(bsz);
                for (var i = 0; i < bsz; i++) { var s = srcs[PickSrc()].S; batch.Add(s.GetExample(rng.NextInt64(s.Count))); }   // VOLUME-WEIGHTED: folder ∝ Count^MixAlpha (see the [mix] log), then an example
                if (!CorpusOnly) { var drained = 0; while (drained < 32 && _inbox.TryDequeue(out var ex2)) { batch.Add(ex2); drained++; } }   // network-fed learning (muted in corpus-only)
                if (batch.Count == 0) break;
                var bt0 = sw.Elapsed.TotalSeconds;
                lock (_write)
                {
                    // GPU trains the batch LOCALLY and writes to the CPU model (source of truth). We're a host by default
                    // but yield to the network between batches (bleed/gossip run on their own timers off the fresh _model),
                    // so hosting no longer means CPU — the GPU does the compute. Fall back to CPU on any GPU error.
                    if (gpu != null)
                        try { loss = gpu.TrainBatch(batch, lr); }
                        catch (Exception e) { log("[train] GPU batch failed → CPU: " + e.Message.Split('\n')[0]); loss = _trainer.TrainBatch(batch, lr, ct); }
                    else loss = _trainer.TrainBatch(batch, lr, ct);   // ct cancels mid-batch → Stop lands after the current example
                    _real = true;   // trained → weights are real
                }
                stepTotal += batch.Count;
                var secs = sw.Elapsed.TotalSeconds;
                var wps = batch.Count / Math.Max(0.01, secs - bt0);
                if (first) { first = false; log("live — what the model generates on your data:"); }
                if (secs - lastSnap >= 3.0) { var snap = Clone(_model); _snapshot = snap; lastSnap = secs; }   // refresh serving snapshot
                report($"epoch {ep + 1}/{epochs} · {secs - epStart:F0}/{epochSecs:F0}s · loss {loss:F3} · {wps:F1} win/s{(Hosting ? $" · {ActiveWorkers} peer(s)" : "")}");   // metrics → status bar
                if (secs - lastSample >= 1.2 && !genBusy)   // CYCLE folders; GENERATE the whole answer (stops at the STOP token) in the BACKGROUND so it never blocks training/Stop
                {
                    lastSample = secs;
                    var name = srcs[sampleTick++ % srcs.Count].Name;
                    const int preview = 200;   // chars generated for the progress preview (was 18); raise toward PrismSpec.Context to watch longer reproductions
                    string prompt = "", want = "";
                    if (name == "text")
                    {
                        var wctx = corpus.GetExample(rng.NextInt64(Math.Max(1, corpus.Count))).Ctx;
                        var split = Math.Max(1, wctx.Length - preview);
                        prompt = _v.Decode(wctx[..split]); want = Clean(_v.Decode(wctx[split..]));   // "want" = the real next chars that follow the window
                    }
                    // KEEP the prompt RAW (newlines + trailing "prism: " priming intact) — the model trained on this exact
                    // form, so the sample must feed it the same. (The old code did .Replace("\n"," ").Trim() here, which is a
                    // DISPLAY transform; applying it to the generation input primed the model off-distribution → ":::"/"===".)
                    else if (name == "chat") { var cps = PairSource.ChatPairs(chatText); if (cps.Count > 0) { var qp = cps[rng.Next(cps.Count)]; prompt = qp.Prompt; want = qp.Target; } }
                    else if (name == "group") { var gps = GroupChat.Pairs(GroupContext); if (gps.Count > 0) { var qp = gps[rng.Next(gps.Count)]; prompt = qp.Prompt; want = qp.Target; } }   // sample the REAL group buffer, not data/pairs
                    else if (name == "peer") { var pd = _peerDistill.ToArray(); if (pd.Length > 0) { var qp = pd[rng.Next(pd.Length)]; prompt = qp.Ctx; want = qp.Reply; } }
                    else if (name == "qa-chat") { var qaAll = PairSource.ReadFolders(PairsDir, GossipDir); if (qaAll.Count > 0) { var fr = GroupChat.AsChat(new[] { qaAll[rng.Next(qaAll.Count)] }).First(); prompt = fr.Prompt; want = fr.Target; } }   // show the chat-framed prompt the model actually answers
                    else { var ls = RawLines(name == "gossip" ? GossipDir : PairsDir); if (ls.Length > 0) { var l = ls[rng.Next(ls.Length)]; var tb = l.IndexOf('\t'); prompt = l[..tb]; want = l[(tb + 1)..].Trim(); } }
                    if (prompt.Length > 0)
                    {
                        var gen = prompt.EndsWith(" ") ? prompt : prompt + " ";   // SAME trailing-space priming PairSource trains with — so the sample sees what the model REALLY does (matches chat/serve), not an off-distribution prompt
                        genBusy = true; string pr = prompt, gp = gen, wnt = want, nm = name;
                        Task.Run(() => { try { log($"[{nm}·{pr.Length}c ctx] {Clean(Tail(pr))} ▸ \"{Clean(GenAnswer(gp, preview))}\"  (want \"{wnt}\")"); } catch { } finally { genBusy = false; } });
                    }
                }
            }
            if (!ct.IsCancellationRequested) _snapshot = Clone(_model);   // snapshot at each COMPLETED epoch (skip on cancel → faster stop)
        }
        if (!ct.IsCancellationRequested) _snapshot = Clone(_model);
        gpu?.Dispose();
        log($"[train] done — {stepTotal:N0} windows, loss {loss:F3}");
        report($"done · {stepTotal:N0} steps · loss {loss:F3}");
    }

    /// <summary>Serve a peer's query (the networked REPL): a continuation of their prompt + our confidence (top-2
    /// margin on the first token). Reads the lock-free snapshot, so it answers even mid-training.</summary>
    public (double Conf, string Continuation) Serve(string prompt)
    {
        var t0 = prompt ?? ""; var primed = (t0.Length == 0 || t0.EndsWith("\n") ? t0 : t0 + "\n") + GroupChat.AiTag;   // NATURAL prime: append our reply slot "prism: " to the recent context. NO swap — the swap belongs in ingestion (GroupChat.Pairs), so serve hits the well-trained "user: X\nprism:" pattern
        var ids = _v.Encode(primed);
        var len = Math.Min(ids.Length, PrismSpec.Context);                        // left-aligned last-Context window (causal, natural positions)
        var ctx = len > 0 ? ids[(ids.Length - len)..] : new[] { CharVocab.Pad };
        return (Answer(ctx).Margin, Sample(primed, 160));   // continuation of the human slot = the model's reply
    }

    /// <summary>A pair bled in from a neighbour → gossip folder (deduped + capped), trained next epoch.</summary>
    public void AddGossipPair(string prompt, string target) => _gossip.Add(new[] { (prompt, target) });

    // ── GROUP CHAT: a shared network conversation, saved rolling + context-capped, trained as (context → next HUMAN turn) ──
    public string GroupContext { get { lock (_group) return _groupBuf; } }

    /// <summary>Append a turn to the rolling group log + persist it, capped at the context window (oldest chars trimmed →
    /// the model only ever trains on the MOST RECENT context). human=true is a real person's turn (a training TARGET);
    /// human=false is a model turn (context only — we never train toward AI-generated text).</summary>
    public void AppendGroup(bool human, string text)
    {
        text = Clean(text).Trim(); if (text.Length == 0) return;
        lock (_group)
        {
            _groupBuf = CapRecent(_groupBuf + (human ? GroupChat.HumanTag : GroupChat.AiTag) + text + "\n", PrismSpec.Context);
            try { File.WriteAllText(GroupFile, _groupBuf); } catch { }
        }
    }

    /// <summary>Seed the rolling group buffer from another cell's LIVE context — used when a fresh sub-node starts with an
    /// empty group log, so RandomPrompt (peer queries) and the group lane have something to work with from the first tick
    /// instead of waiting for network group traffic to accumulate. No-op once our own buffer has content (never clobbers).</summary>
    public void SeedGroup(string context)
    {
        if (string.IsNullOrEmpty(context)) return;
        lock (_group)
        {
            if (_groupBuf.Length > 0) return;
            _groupBuf = CapRecent(context, PrismSpec.Context);
            try { File.WriteAllText(GroupFile, _groupBuf); } catch { }
        }
    }

    /// <summary>This node's group reply: continue the conversation in a HUMAN voice (all the model is trained to do) —
    /// prime the human slot on the recent context and generate ONE turn (stops at the STOP token). Posted as an AI turn.</summary>
    public string GroupReply(string transcript) { var t = Tail(transcript ?? "", PrismSpec.Context); return GenerateCached(_snapshot, (t.Length == 0 || t.EndsWith("\n") ? t : t + "\n") + GroupChat.AiTag, 80); }   // NATURAL prime (no swap — see Serve)

    /// <summary>Absorb a neighbour's bled pair. Always persists it to gossip (durable, trained next epoch). PLUS the
    /// swarmlearn "backprop-on-wrong" result: a bled pair is LABELLED, so if our current model does NOT already produce
    /// it, enqueue its examples for IMMEDIATE gradient uptake (learned this batch, not next epoch). Gated on wrong-only —
    /// re-learning what we already know wastes a step and, per the experiment, indiscriminate absorption dents our own
    /// specialty (interference). No-op on the fast path if we're not training (the inbox is drained by the writer).</summary>
    public void AbsorbPair(string prompt, string target)
    {
        AddGossipPair(prompt, target);   // durable, gossiped
        AbsorbExamples(prompt, target);  // + immediate backprop-on-wrong
    }

    /// <summary>Backprop-on-wrong on a single labelled pair (NO gossip): if our model doesn't already produce it, enqueue
    /// its examples for a bounded gradient step. Shared by <see cref="AbsorbPair"/> and <see cref="AbsorbChat"/>.</summary>
    void AbsorbExamples(string prompt, string target)
    {
        if (string.IsNullOrEmpty(target)) return;
        try
        {
            var ps = new PairSource(PrismSpec.Context, new[] { (prompt, target) }, _v);
            if (ps.Count == 0) return;
            var (c0, t0) = ps.GetExample(0);
            if (_snapshot.Predict(c0) == t0) return;                       // already known → skip (avoid interference)
            for (long i = 0; i < ps.Count; i++) { var (c, t) = ps.GetExample(i); SubmitExample(c, t); }
        }
        catch { }
    }

    /// <summary>A peer QUERIED us (REPL escalation or group round-robin) — the query carries THEIR human turns, real human
    /// data. Learn from those: backprop-on-wrong on each (context → human turn) pair in the query context. So when nodes
    /// query each other, the one ASKED trains (this) while the one ASKING distils the reply (<see cref="LearnFromPeer"/>) —
    /// both sides learn, and human data spreads through the swarm. NOT gossiped (chat/group data isn't).</summary>
    public void AbsorbChat(string context)
    {
        if (string.IsNullOrWhiteSpace(context)) return;
        try { foreach (var (p, t) in GroupChat.Pairs(context)) AbsorbExamples(p, t); } catch { }   // FULL context — the anchor doesn't reply, so its core is free to learn from every human turn
    }

    /// <summary>Drain up to <paramref name="max"/> network-fed examples (bled pairs this node got wrong, enqueued by
    /// <see cref="AbsorbPair"/>) and take ONE gradient step on them — the "learn from being chattered to" path for a node
    /// with NO curriculum loop (the anchor). Bounded + single-step so it can't pin a weak box; refreshes the serving
    /// snapshot so answers reflect what it just learned. Returns how many examples it trained on.</summary>
    public int AbsorbStep(int max = 6, double lr = 1e-3)
    {
        var batch = new List<(int[] Ctx, int Target)>(max);
        while (batch.Count < max && _inbox.TryDequeue(out var ex)) batch.Add(ex);
        if (batch.Count == 0) return 0;
        lock (_write) { _trainer.TrainBatch(batch, lr); _real = true; _snapshot = Clone(_model); }
        return batch.Count;
    }

    readonly ConcurrentQueue<(string Ctx, string Reply)> _peerDistill = new();
    /// <summary>Distil a peer's reply to one of OUR OWN chat contexts (see <see cref="RandomPrompt"/>): train it EXACTLY
    /// like our own chat — <c>context → reply</c> — so the personality flavour the peer grew from ITS chat context blends
    /// into ours. Held in memory, capped; mixed into training as the "peer" source (same window/format as the chat source).</summary>
    public void LearnFromPeer(string context, string reply)
    {
        if (string.IsNullOrWhiteSpace(context) || string.IsNullOrWhiteSpace(reply)) return;
        _peerDistill.Enqueue(((context.EndsWith("\n") ? context : context + "\n") + GroupChat.AiTag, reply));   // NATURAL context + our reply slot, matching natural serve (no swap)
        while (_peerDistill.Count > 2000) _peerDistill.TryDequeue(out _);
    }

    /// <summary>Concatenated text of our own chat context file(s) — the personality we've grown from talking.</summary>
    string ReadChat() => Directory.Exists(ChatDir) ? string.Concat(Directory.EnumerateFiles(ChatDir, "*.txt").OrderBy(f => f).Select(File.ReadAllText)) : "";

    /// <summary>A random local pair to bleed to neighbours (confident teaching signal). Cached on first use.</summary>
    public (string Prompt, string Target)? RandomPair(Random rng)
    {
        _pairCache ??= Directory.EnumerateFiles(PairsDir, "*.tsv").SelectMany(File.ReadLines).Where(l => l.Contains('\t')).ToArray();
        if (_pairCache.Length == 0) return null;
        var l = _pairCache[rng.Next(_pairCache.Length)]; var t = l.IndexOf('\t');
        return (l[..t], l[(t + 1)..]);
    }

    /// <summary>A random conversational context to ask a peer "what would you say here?" — drawn from BOTH our REPL chat
    /// AND the group chat (a prompt ending "user: "). We distil the peer's reply; the peer, being asked, learns from our
    /// human turns (<see cref="AbsorbChat"/>). Null if we have no chat yet.</summary>
    public string? RandomPrompt(Random rng)
    {
        var t = (GroupContext.Length > 0 && rng.Next(2) == 0) ? GroupContext : ReadChat();   // draw from the group OR the REPL chat
        var lines = t.Replace("\r", "").Split('\n').Where(l => l.Length > 0).ToArray();
        if (lines.Length == 0) return null;
        return string.Join("\n", lines.Take(1 + rng.Next(lines.Length))) + "\n";   // a NATURAL prefix ending after a turn — the peer's Serve primes it naturally (append AiTag, no swap)
    }

    /// <summary>Feed a labelled example to the writer (bled pair from a neighbour, or backprop-on-wrong from a swarm
    /// query). Trained into the model on the next batch — the single-writer path, so it never races.</summary>
    public void SubmitExample(int[] ctx, int target) { if (_inbox.Count < 8192) _inbox.Enqueue((ctx, target)); }   // bounded: the writer drains it only while training

    // ════════ READERS — lock-free, serve off the snapshot (never blocks training) ════════
    public string Sample(string prompt, int n) => GenerateCached(_snapshot, prompt, n);   // causal KV-cache generation (O(T)/token), stops at STOP

    /// <summary>Next-token prediction + confidence (top-2 logit margin) off the snapshot — for the networked REPL's
    /// "am I sure, or should I escalate to the swarm?" gate.</summary>
    public (int Pred, double Margin) Answer(int[] ctx)
    {
        var lg = _snapshot.LogitsFor(ctx);
        int i1 = 0; for (var i = 1; i < lg.Length; i++) if (lg[i] > lg[i1]) i1 = i;
        var second = double.NegativeInfinity; for (var i = 0; i < lg.Length; i++) if (i != i1 && lg[i] > second) second = lg[i];
        return (i1, lg[i1] - second);
    }

    // ════════ CONTROL — serialize with the writer ════════
    /// <summary>Persist the model — CRASH-SAFE and CLOBBER-SAFE. (1) Refuses to overwrite an existing checkpoint with a
    /// model that was never loaded or trained (<see cref="_real"/> = false) — that was the data-loss bug: a corrupt/failed
    /// load booted a FRESH model and the autosave then wiped the good weights. (2) Writes fully to a .tmp then ATOMICALLY
    /// swaps it in (File.Replace), so an interrupted save can never leave a truncated/corrupt checkpoint (which was what
    /// made the next load fail in the first place). (3) Keeps the prior checkpoint as .bak, so any overwrite is recoverable.</summary>
    public void Save(string path)
    {
        lock (_write)
        {
            if (!_real && File.Exists(path)) return;   // never clobber a good checkpoint with an untrained fresh model
            var tmp = path + ".tmp";
            using (var w = new BinaryWriter(File.Create(tmp))) { w.Write(Signature); _model.Save(w); }
            if (File.Exists(path)) File.Replace(tmp, path, path + ".bak");   // atomic swap + one-deep backup
            else File.Move(tmp, path);
        }
    }

    /// <summary>Wipe to a fresh model (current spec init) and delete the checkpoint. Training data (text/pairs/chat) is
    /// KEPT — retrain to rebuild. Used by the Studio "Reset model" button, e.g. after a spec change orphans old weights.</summary>
    public void Reset(string checkpointPath)
    {
        lock (_write) { _model.ReinitFrom(PrismSpec.NewModel(Seed)); _snapshot = Clone(_model); }
        try { if (File.Exists(checkpointPath)) File.Delete(checkpointPath); } catch { }
    }
    public bool Load(string path)
    {
        if (!File.Exists(path)) return false;
        lock (_write)
        {
            try
            {
                using var r = new BinaryReader(File.OpenRead(path));
                var sig = r.ReadString();
                if (sig == Signature) { if (!_model.Load(r)) return false; }
                else
                {
                    var old = PrismSpec.Parse(sig);                                   // older/other spec on disk
                    if (old is null || !PrismSpec.CanUpgradeFrom(old)) return false;  // hard fork → caller starts fresh
                    if (!_model.LoadUpgrade(r, old.Context)) return false;            // UPGRADE-IN-PLACE (zero-pad Shifts/Context)
                }
                _snapshot = Clone(_model);
                _real = true;   // successfully loaded real weights → safe to autosave over the source
                return true;
            }
            catch { return false; }
        }
    }

    public bool Hosting => _host?.Hosting ?? false;
    public int ActiveWorkers => _host?.ActiveWorkers ?? 0;
    // Peers CAN co-train through the relay again. Safe now: with 0 peers the host trains each batch locally & immediately
    // (no buffer, no block), and DoRound honors the cancel token — so an empty room never freezes training and Stop lands.
    public string EnableHosting(Action<string>? onEvent = null)
    {
        var host = new MqttRelayHost(_model);
        var code = host.StartRelay(onEvent);            // broker connect OUTSIDE the write lock — never blocks training
        lock (_write) { _host = host; _trainer = host; }
        return code;
    }
    public void DisableHosting() { lock (_write) { _host?.StopRelay(); _host = null; _trainer = new PrismTrainer(_model); } }

    // ---- tiny elastic weight-averaging over the bleed channel: exchange a small random slice of params so peers slowly converge ----
    static byte[] SaveBytes(AlgFormer m) { using var ms = new MemoryStream(); var w = new BinaryWriter(ms); m.Save(w); w.Flush(); return ms.ToArray(); }

    /// <summary>A small random slice of the model's params (start index + values) — tiny payload for a bleed.</summary>
    public (int Start, double[] Vals)? WeightSlice(Random rng, int count)
    {
        lock (_write)
        {
            var b = SaveBytes(_model); var n = (b.Length - 16) / 8;   // doubles after the 4-int header
            if (n <= count) return null;
            var start = rng.Next(n - count); var vals = new double[count];
            for (var i = 0; i < count; i++) vals[i] = BitConverter.ToDouble(b, 16 + (start + i) * 8);
            return (start, vals);
        }
    }

    /// <summary>Nudge our params at [start..] toward a peer's slice by alpha (elastic average). Over many bleeds the
    /// whole model converges. Runs under the single-writer lock; call off the broker thread.</summary>
    public void MergeWeightSlice(int start, double[] vals, double alpha)
    {
        lock (_write)
        {
            var b = SaveBytes(_model); var n = (b.Length - 16) / 8;
            for (var i = 0; i < vals.Length && start + i >= 0 && start + i < n; i++)
            {
                var mine = BitConverter.ToDouble(b, 16 + (start + i) * 8);
                BitConverter.GetBytes((1 - alpha) * mine + alpha * vals[i]).CopyTo(b, 16 + (start + i) * 8);
            }
            using var r = new BinaryReader(new MemoryStream(b)); _model.Load(r);
            _model.RestoreFrozen(Seed);   // the byte-average may have nudged the frozen number/identity faces — pin them back to exact
            _snapshot = Clone(_model);
        }
    }
}
