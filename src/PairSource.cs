// Copyright (c) 2026 Dongyang Stephen Chen, trading as Evaluated Applications. All rights reserved.
// Source-available for study and verification; commercial use, redistribution, or derivatives require permission. See LICENSE.

using System.Security.Cryptography;
using System.Text;

namespace PrismFormer;

/// <summary>
/// Shared character vocabulary — printable ASCII (32..126, 95 symbols). The ONE tokenization that lets a single
/// model serve corpus, pairs, and gossip alike (see STUDIO.md). Out-of-range chars fold to space.
/// </summary>
public sealed class CharVocab
{
    public const int Lo = 32, Hi = 126;
    public const int Printable = Hi - Lo + 1;                     // 95 printable ASCII
    public const int End = Printable;                            // 95 — end-of-turn / STOP token; the model learns to emit it so the REPL knows to stop
    public const int N = Printable + 1;                          // 96 — the CHAR base vocab (printable + STOP). Subword tokens (if any) are appended AFTER this.
    public const int Pad = 0;                                    // space — left-pad for short contexts

    // Shared, deterministic subword extension (n-grams len 2..4) loaded from the committed embedded resource. EMPTY =>
    // pure char level (Encode == char ids, Total == 96). Static so every CharVocab instance and every node agree.
    static readonly SubwordVocab _sw = new(SubwordTable.List);
    public static int Total => _sw.Size;                         // 96 + subword count — THE model vocab (PrismSpec.Vocab uses this)
    public int Size => Total;

    public int Id(char c) => c >= Lo && c <= Hi ? c - Lo : 0;   // printable → 0..94; everything else (incl newline) → space. STOP is appended EXPLICITLY (Q&A/chat), never from corpus text.
    public char Chr(int id) => id == End ? '\n' : (char)(Lo + Math.Clamp(id, 0, Printable - 1));
    public string Symbol(int id) => _sw.Symbol(id);             // token id → its literal text (char OR subword); for decode AND codec seeding
    public int[] Encode(string s) => _sw.Encode(s);            // greedy longest-match subword tokenisation (byte-identical to char 1:1 when the list is empty)
    public string Decode(IEnumerable<int> ids) => _sw.Decode(ids);
}

/// <summary>
/// CORPUS lane (see STUDIO.md): continuous text, sliced into dense (context -> next char) windows. Every position is
/// supervised, and windows RESET at each file/segment boundary so they never span unrelated text. Trains fluency.
/// </summary>
public sealed class CorpusSource : IJobSource
{
    readonly int _ctx;
    readonly List<int[]> _segs = new();
    readonly long[] _cum;                 // prefix sum of per-segment window counts
    public CharVocab Vocab { get; }

    public CorpusSource(int ctx, IEnumerable<string> segments, CharVocab? vocab = null)
    {
        Vocab = vocab ?? new CharVocab(); _ctx = ctx;
        foreach (var s in segments) { if (string.IsNullOrEmpty(s)) continue; var a = Vocab.Encode(s); if (a.Length > ctx) _segs.Add(a); }
        _cum = new long[_segs.Count + 1];
        for (var i = 0; i < _segs.Count; i++) _cum[i + 1] = _cum[i] + (_segs[i].Length - _ctx);
    }

    public static CorpusSource FromFolders(int ctx, CharVocab? vocab, params string[] folders)
    {
        var segs = new List<string>();
        foreach (var f in folders)
            if (f != null && Directory.Exists(f))
                foreach (var file in Directory.EnumerateFiles(f, "*.txt", SearchOption.AllDirectories).OrderBy(x => x))
                    segs.Add(System.Text.RegularExpressions.Regex.Replace(File.ReadAllText(file), @"\\x[0-9A-Fa-f]{2}|\\u[0-9A-Fa-f]{4}", " "));   // strip literal byte-escape junk ("\xc3\xa9") some corpora carry
        return new CorpusSource(ctx, segs, vocab);
    }

    public long Count => _cum[^1];
    public (int[] Ctx, int Target) GetExample(long index)
    {
        int seg = UpperBound(_cum, index) - 1; var o = (int)(index - _cum[seg]);
        var a = _segs[seg]; var c = new int[_ctx]; Array.Copy(a, o, c, 0, _ctx);
        return (c, a[o + _ctx]);
    }
    static int UpperBound(long[] cum, long v) { int lo = 0, hi = cum.Length; while (lo < hi) { var m = (lo + hi) / 2; if (cum[m] <= v) lo = m + 1; else hi = m; } return lo; }
}

/// <summary>
/// PAIR lane (see STUDIO.md): independent <c>prompt -> target</c> examples. Each pair expands into one example per
/// TARGET char (teacher forcing over the target only); the context is the prompt+target-so-far, left-padded, and
/// NEVER spans into another pair. Trains behaviour / instruction-following. Line format: <c>prompt\ttarget</c>.
/// </summary>
public sealed class PairSource : IJobSource
{
    readonly int _ctx;
    readonly List<(int[] Full, int PromptLen)> _pairs = new();
    readonly long[] _cum;                 // prefix sum of per-pair target lengths (= examples per pair)
    public CharVocab Vocab { get; }

    public PairSource(int ctx, IEnumerable<(string Prompt, string Target)> pairs, CharVocab? vocab = null)
    {
        Vocab = vocab ?? new CharVocab(); _ctx = ctx;
        foreach (var (p, t) in pairs)
        {
            if (string.IsNullOrEmpty(t)) continue;
            var prompt = p.EndsWith(" ") ? p : p + " ";   // guarantee a space between prompt and target so words don't mush ("frigidfreezing")
            // Encode prompt and target SEPARATELY. PromptLen must be the prompt's TOKEN span (the index in `full` where the
            // target begins), NOT its character count. Under the char vocab those were equal; under the subword vocab they
            // are NOT — storing chars made Count = Σ(Full.Length − PromptLen) go NEGATIVE for long-prompt/short-target pairs,
            // sinking whole lanes to Count≤0 so they were silently DROPPED, and mis-aligning GetExample's predict index for
            // the ones that survived. Separate encoding also keeps the prompt|target seam tokenised the way generation primes
            // it at inference (prompt encoded on its own), so train and serve agree.
            var pTok = Vocab.Encode(prompt);
            var tTok = Vocab.Encode(t);
            var full = new int[pTok.Length + tTok.Length + 1];
            Array.Copy(pTok, full, pTok.Length);
            Array.Copy(tTok, 0, full, pTok.Length, tTok.Length);
            full[^1] = CharVocab.End;   // explicit STOP token after the answer (Q&A + chat) → REPL stops here; corpus stays pure next-token
            _pairs.Add((full, pTok.Length));
        }
        _cum = new long[_pairs.Count + 1];
        for (var i = 0; i < _pairs.Count; i++) _cum[i + 1] = _cum[i] + (_pairs[i].Full.Length - _pairs[i].PromptLen);
    }

    /// <summary>Read raw (prompt, target) tsv/pairs lines from folders — the shared reader for <see cref="FromFolders"/> and
    /// for chat-wrapping the same Q&amp;A (<see cref="GroupChat.AsChat"/>).</summary>
    public static List<(string Prompt, string Target)> ReadFolders(params string[] folders)
    {
        var pairs = new List<(string Prompt, string Target)>();
        foreach (var f in folders)
            if (f != null && Directory.Exists(f))
                foreach (var file in Directory.EnumerateFiles(f, "*", SearchOption.AllDirectories).Where(x => x.EndsWith(".tsv") || x.EndsWith(".pairs")).OrderBy(x => x))
                    foreach (var line in File.ReadLines(file))
                    {
                        var tab = line.IndexOf('\t'); if (tab <= 0 || tab == line.Length - 1) continue;
                        pairs.Add((line[..tab], line[(tab + 1)..]));   // prompt | target, separator dropped (prompt's own ending is the boundary)
                    }
        return pairs;
    }

    public static PairSource FromFolders(int ctx, CharVocab? vocab, params string[] folders) => new(ctx, ReadFolders(folders), vocab);

    /// <summary>Progressive conversational pairs from a "user: …\nprism: …" transcript — UNIFIED with the group chat:
    /// human ("user:") turns are the TARGETS (the model learns to produce human text), the model's ("prism:") turns are
    /// context only (never trained toward). Prompts end "user: " so the REPL and the group chat share one human-target
    /// slot. STOP is appended per target by <see cref="PairSource"/>. (Legacy "you:"/"ai:" transcripts still parse.)</summary>
    public static PairSource FromChat(int ctx, CharVocab? vocab, string transcript) => new(ctx, ChatPairs(transcript), vocab);

    public static List<(string Prompt, string Target)> ChatPairs(string transcript) => GroupChat.Pairs(transcript);

    public long Count => _cum[^1];
    public (int[] Ctx, int Target) GetExample(long index)
    {
        int p = UpperBound(_cum, index) - 1; var k = (int)(index - _cum[p]);
        var (full, plen) = _pairs[p];
        var predict = plen + k;                 // index in `full` of the target char we predict
        // LEFT-ALIGNED, natural positions (token i at position i) — the CAUSAL convention, matching KV-cache generation
        // which primes the prompt at positions 0..len-1 and predicts the next token. Take the last _ctx tokens of the
        // prefix when it overflows the window (rebased to 0), exactly as generation slides. No left-pad.
        var len = Math.Min(predict, _ctx);
        var c = new int[len];
        Array.Copy(full, predict - len, c, 0, len);
        return (c, full[predict]);
    }
    static int UpperBound(long[] cum, long v) { int lo = 0, hi = cum.Length; while (lo < hi) { var m = (lo + hi) / 2; if (cum[m] <= v) lo = m + 1; else hi = m; } return lo; }
}

/// <summary>Turn a GROUP-CHAT transcript into training pairs — <c>(conversation so far → what a HUMAN said next)</c>.
/// The log is lines "<c>user: …</c>" (a human turn) and "<c>ai: …</c>" (a model turn). We emit one pair per HUMAN turn:
/// prompt = everything before it + "<c>user: </c>", target = the human's text. AI turns are context ONLY, never targets —
/// the group discussion is chunked into many key→value examples from one window, and the model is only ever taught to
/// produce HUMAN-generated text (we don't trust AI text as the goal). Feeds <see cref="PairSource"/> like the REPL chat,
/// but with no identity switch: it doesn't impersonate one speaker, it predicts the next human contribution.</summary>
public static class GroupChat
{
    public const string HumanTag = "user: ", AiTag = "prism: ";   // human = user (ambiguous), AI = prism — shared by the group chat AND the REPL

    // NAME SWAP: the model speaks as "prism:", and it produces HUMAN turns (it assumes the human identity). So a human
    // turn is the TARGET under the MODEL's own slot "prism: ", and in the swapped context humans become "prism:" while AI
    // turns become the "user:" interlocutor. Why it's sound even when the model talks crap: the AI's own past output —
    // nonsense included — is just context the model reacts to; the HUMAN's genuine response to it is the valid target, so
    // the model learns to react like a human (e.g. surprise at nonsense). Transcripts stay NATURAL everywhere else; the
    // swap lives only here (training) and in generation priming (see StudioModel.Serve/GroupReply → GroupChat.Swap).
    public static List<(string Prompt, string Target)> Pairs(string transcript)
    {
        var pairs = new List<(string, string)>();
        var ctx = new StringBuilder();
        foreach (var raw in (transcript ?? "").Replace("\r", "").Split('\n'))
        {
            var (ok, human, text) = Role(raw);
            if (!ok) continue;
            if (human)
            {
                // A no-context opener (ctx still empty) would teach "produce <line> from NOTHING" — which never occurs at
                // serve (the model always has the interlocutor's turn in front of it) and collapses to the marginal = the
                // most frequent opener ("hi"/"hello"), the attractor you can't get it out of. Only emit once there's real
                // preceding context to condition the reply on. A context-free line teaches nothing conditional, only the mode.
                if (text.Length > 0 && ctx.Length > 0) pairs.Add((ctx + AiTag, text));   // human turn → TARGET, under the model's slot "prism: "
                ctx.Append(AiTag).Append(text).Append('\n');           // human → "prism:" in the swapped context
            }
            else ctx.Append(HumanTag).Append(text).Append('\n');       // AI turn → "user:" interlocutor (context only)
        }
        return pairs;
    }

    /// <summary>Frame raw Q&amp;A pairs the NATURAL way the REPL and group chat serve them — <c>"user: Q\nprism: " → A</c> —
    /// so PRISM produces the answer at inference. The swap lives in INGESTION (chat logs → pairs, see <see cref="Pairs"/>),
    /// NOT at serve: Serve just appends <c>AiTag</c> to the recent context, so the model generates its reply after a
    /// <c>user:</c> turn — the well-trained pattern. Already-framed pairs (e.g. a gossiped chat pair) are skipped so they
    /// aren't double-wrapped. The answer is ground-truth, so training toward it under <c>prism:</c> is safe (no collapse).</summary>
    public static IEnumerable<(string Prompt, string Target)> AsChat(IEnumerable<(string Prompt, string Target)> qa)
        => qa.Where(p => !Framed(p.Prompt)).Select(p => (HumanTag + p.Prompt.Trim() + "\n" + AiTag, p.Target));

    static bool Framed(string p) { var t = p.TrimStart(); return t.StartsWith("user:") || t.StartsWith("prism:") || t.StartsWith("you:") || t.StartsWith("ai:"); }

    /// <summary>Relabel a NATURAL transcript into the model's identity-swapped view (human↔prism) — the mirror of
    /// <see cref="Pairs"/>, used to prime generation so train and infer share one view. Everything else stays natural.</summary>
    public static string Swap(string transcript)
    {
        var sb = new StringBuilder();
        foreach (var raw in (transcript ?? "").Replace("\r", "").Split('\n'))
        {
            var (ok, human, text) = Role(raw);
            if (ok) sb.Append(human ? AiTag : HumanTag).Append(text).Append('\n');
        }
        return sb.ToString();
    }

    // Classify a transcript line. Canonical: "user:"/"ai:". Legacy REPL "you:"/"prism:" still parse (map to human/ai).
    static (bool ok, bool human, string text) Role(string line)
    {
        if (line.StartsWith("user: ")) return (true, true, line[6..].Trim());
        if (line.StartsWith("you: ")) return (true, true, line[5..].Trim());
        if (line.StartsWith("ai: ")) return (true, false, line[4..].Trim());
        if (line.StartsWith("prism: ")) return (true, false, line[7..].Trim());
        return (false, false, "");
    }
}

/// <summary>Union several sources (corpus + pairs + gossip) into one index-addressable stream for the trainer.</summary>
public sealed class MixSource : IJobSource
{
    readonly IJobSource[] _src; readonly long[] _cum;
    public MixSource(params IJobSource[] sources) { _src = sources.Where(s => s.Count > 0).ToArray(); _cum = new long[_src.Length + 1]; for (var i = 0; i < _src.Length; i++) _cum[i + 1] = _cum[i] + _src[i].Count; }
    public long Count => _cum[^1];
    public (int[] Ctx, int Target) GetExample(long index) { int s = UpperBound(_cum, index) - 1; return _src[s].GetExample(index - _cum[s]); }
    static int UpperBound(long[] cum, long v) { int lo = 0, hi = cum.Length; while (lo < hi) { var m = (lo + hi) / 2; if (cum[m] <= v) lo = m + 1; else hi = m; } return lo; }
}

/// <summary>
/// The <c>gossip/</c> folder as a live inbox: neighbours append distilled pairs here and they become a training
/// source next rebuild. Deduplicated and capacity-capped (oldest trimmed) so gossip can't explode — the first line
/// of trust, alongside per-pair provenance. Writes <c>prompt\ttarget</c> lines the PairSource reads.
/// </summary>
public sealed class GossipInbox
{
    readonly string _file;
    readonly int _cap;
    readonly HashSet<string> _seen = new();
    readonly LinkedList<string> _lines = new();
    readonly object _lock = new();
    public string Folder { get; }

    public GossipInbox(string folder, int cap = 50_000)
    {
        Folder = folder; _cap = cap;
        Directory.CreateDirectory(folder);
        _file = Path.Combine(folder, "gossip.tsv");
        if (File.Exists(_file)) foreach (var l in File.ReadLines(_file)) { if (l.Length == 0) continue; if (_seen.Add(Key(l))) _lines.AddLast(l); }
        Trim();
    }

    /// <summary>Add received pairs. Returns how many were new after dedup. Persists the capped, deduped set.</summary>
    public int Add(IEnumerable<(string Prompt, string Target)> pairs, string? from = null)
    {
        var added = 0;
        lock (_lock)
        {
            foreach (var (p, t) in pairs)
            {
                // clean prompt\ttarget so PairSource reads it directly (provenance kept out of the trainable line for now)
                var line = $"{Clean(p)}\t{Clean(t)}";
                if (!_seen.Add(Key(line))) continue;
                _lines.AddLast(line); added++;
            }
            Trim();
            File.WriteAllLines(_file, _lines);
        }
        return added;
    }

    public int Count { get { lock (_lock) return _lines.Count; } }
    void Trim() { while (_lines.Count > _cap) { var g = _lines.First!.Value; _lines.RemoveFirst(); _seen.Remove(Key(g)); } }
    static string Clean(string s) => s.Replace('\t', ' ').Replace('\n', ' ').Replace('\r', ' ').Trim();
    static string Key(string line) => Convert.ToHexString(MD5.HashData(Encoding.UTF8.GetBytes(line)));
}
