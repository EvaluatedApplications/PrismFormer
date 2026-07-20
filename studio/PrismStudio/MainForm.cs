using System.Diagnostics;
using PrismGym;
using PrismFormer;

namespace PrismStudio;

/// <summary>
/// Prism Studio (see STUDIO.md) — two tabs. MODEL: one frozen-spec model, trained on a folder of pair/corpus files
/// (anyone can drop more, no code), plus a REPL to talk to it. NETWORK: join a swarm or host over the free relay.
/// Gym + BabyLM are gone as live tabs — they're now the "Generate starter data" button, which writes starter files.
/// </summary>
public sealed class MainForm : Form
{
    private static string PrismDir()
    {
        var d = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Prism");
        Directory.CreateDirectory(d); return d;
    }
    private readonly string _dataDir = Path.Combine(PrismDir(), "data");                 // persists across rebuilds, alongside the checkpoint (NOT bin/data, which every rebuild wipes)
    private readonly string _savePath = Path.Combine(PrismDir(), "prism.bin");           // trained weights persist across rebuilds

    /// <summary>Copy the data that ships with the app (bin/data = the repo's studio/PrismStudio/data, via Content) into
    /// the persistent AppData data folder on EVERY launch, OVERWRITING — so the repo's training data is always
    /// authoritative and any edit to it takes effect next run. RUNTIME lanes (gossip/chat/group) are NEVER touched — they
    /// accumulate at runtime (bled pairs, conversations), so overwriting them from the repo seed would wipe your data.
    /// Files you add that the repo doesn't ship are left alone.</summary>
    static readonly string[] RuntimeLanes = { "gossip", "chat", "group" };   // accumulate at runtime → keep, never overwrite from the repo
    private void RefreshRepoData()
    {
        try
        {
            var shipped = Path.Combine(AppContext.BaseDirectory, "data");
            if (!Directory.Exists(shipped)) return;
            foreach (var src in Directory.EnumerateFiles(shipped, "*", SearchOption.AllDirectories))
            {
                var rel = Path.GetRelativePath(shipped, src);
                var lane = rel.Split(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar)[0];
                if (RuntimeLanes.Contains(lane)) continue;   // don't clobber gossip/chat/group — that's your runtime data
                var dst = Path.Combine(_dataDir, rel);
                Directory.CreateDirectory(Path.GetDirectoryName(dst)!);
                File.Copy(src, dst, overwrite: true);
            }
        }
        catch { }
    }
    private readonly StudioModel _model;   // self-synchronizing (single-writer + snapshot); no external lock needed
    private long _paramCount;              // cached — recomputing it allocates a full gradient, must NOT run per status update
    private CancellationTokenSource? _trainCts;
    private SwarmChatter? _chatter;
    private System.Threading.Timer? _bleedTimer;
    private const double EscalateMargin = 2.0;   // REPL: below this local confidence, ask the swarm
    private readonly string _chatFile;           // full transcript → reload AND parsed into progressive (conversation → your reply) training pairs
    private string _transcript = "";

    // ---- Model tab ----
    private readonly Button _trainBtn = new() { Text = "▶ Train", Width = 90 };
    private System.Threading.Timer? _mainSaver;   // periodic checkpoint during the (now open-ended) training run
    private readonly CheckBox _autoClear = new() { Text = "auto-clear/hr", AutoSize = true, Checked = true, Padding = new Padding(6, 6, 0, 0) };
    private readonly CheckBox _corpusOnly = new() { Text = "corpus-only", AutoSize = true, Checked = false, Padding = new Padding(6, 6, 0, 0) };   // train on data/text alone (mute pairs/chat/group + inbox)
    private System.Threading.Timer? _autoClearTimer;
    private readonly Label _status = new() { Dock = DockStyle.Fill, TextAlign = ContentAlignment.MiddleLeft, Padding = new Padding(8, 0, 0, 0), Font = new Font("Consolas", 9) };
    private readonly TextBox _log = new() { Multiline = true, ReadOnly = true, ScrollBars = ScrollBars.Vertical, Dock = DockStyle.Fill, Font = new Font("Consolas", 9), BackColor = Color.FromArgb(24, 24, 28), ForeColor = Color.Gainsboro };
    private readonly TextBox _replOut = new() { Multiline = true, ReadOnly = true, ScrollBars = ScrollBars.Vertical, Dock = DockStyle.Fill, Font = new Font("Consolas", 10), BackColor = Color.FromArgb(24, 24, 28), ForeColor = Color.Gainsboro };
    private readonly TextBox _replIn = new() { Dock = DockStyle.Fill, Font = new Font("Consolas", 10) };
    private readonly Button _sendBtn = new() { Text = "Send", Dock = DockStyle.Fill };
    private readonly TextBox _askOut = new() { Multiline = true, ReadOnly = true, ScrollBars = ScrollBars.Vertical, Dock = DockStyle.Fill, Font = new Font("Consolas", 10), BackColor = Color.FromArgb(24, 24, 28), ForeColor = Color.Gainsboro };   // its OWN output — never touches the chat transcript
    private readonly TextBox _askIn = new() { Dock = DockStyle.Fill, Font = new Font("Consolas", 10) };   // "ask a question" — a straight-shot, context-free query
    private readonly Button _askBtn = new() { Text = "Ask", Dock = DockStyle.Fill };
    private readonly Button _saveBtn = new() { Text = "Save", Width = 60 };
    private readonly Button _loadBtn = new() { Text = "Load", Width = 60 };
    private readonly Button _resetBtn = new() { Text = "Reset model", Width = 90 };
    private System.Threading.Timer? _replAnim;
    private volatile bool _replBusy;

    // ---- Network tab ----
    private readonly TextBox _netLog = new() { Multiline = true, ReadOnly = true, ScrollBars = ScrollBars.Vertical, Dock = DockStyle.Fill, Font = new Font("Consolas", 9), BackColor = Color.FromArgb(24, 24, 28), ForeColor = Color.Gainsboro };
    private readonly Label _netStatus = new() { Dock = DockStyle.Fill, TextAlign = ContentAlignment.MiddleLeft, Padding = new Padding(8, 0, 0, 0) };
    private readonly TextBox _groupOut = new() { Multiline = true, ReadOnly = true, ScrollBars = ScrollBars.Vertical, Dock = DockStyle.Fill, Font = new Font("Consolas", 10), BackColor = Color.FromArgb(20, 26, 24), ForeColor = Color.Gainsboro };
    private readonly TextBox _groupIn = new() { Dock = DockStyle.Fill, Font = new Font("Consolas", 10) };
    private readonly Button _groupSend = new() { Text = "Say", Width = 64 };
    private readonly TextBox _hostCode = new() { ReadOnly = true, Width = 320, Font = new Font("Consolas", 9), Text = "(sharing off)" };

    // ---- Sub-node tab: a second in-process cell (its own model / data / checkpoint) you can start/stop, train, and chat with ----
    private readonly string _subDataDir = Path.Combine(PrismDir(), "subnode-data");
    private readonly string _subSavePath = Path.Combine(PrismDir(), "prism-subnode.bin");
    private readonly string _subChatFile = Path.Combine(PrismDir(), "subnode-data", "chat", "chat.txt");   // its REPL transcript persists + reloads (and feeds its "chat" training lane), same as the main cell
    private StudioModel? _subModel;
    private CancellationTokenSource? _subCts;
    private SwarmChatter? _subChatter;
    private System.Threading.Timer? _subSaver;
    private System.Threading.Timer? _subBleedTimer;   // when joined: proactively share pairs/weights + query peers (makes them backprop) and distil their replies
    private string _subTranscript = "";
    private volatile bool _subBusy;
    private readonly Button _subBtn = new() { Text = "▶ Start sub-node", Width = 130 };
    private readonly Label _subStatus = new() { Dock = DockStyle.Fill, TextAlign = ContentAlignment.MiddleLeft, Padding = new Padding(8, 0, 0, 0), Text = "sub-node: stopped" };
    private readonly TextBox _subLog = new() { Multiline = true, ReadOnly = true, ScrollBars = ScrollBars.Vertical, Dock = DockStyle.Fill, Font = new Font("Consolas", 9), BackColor = Color.FromArgb(24, 24, 28), ForeColor = Color.Gainsboro };
    private readonly TextBox _subReplOut = new() { Multiline = true, ReadOnly = true, ScrollBars = ScrollBars.Vertical, Dock = DockStyle.Fill, Font = new Font("Consolas", 10), BackColor = Color.FromArgb(24, 24, 28), ForeColor = Color.Gainsboro };
    private readonly TextBox _subReplIn = new() { Dock = DockStyle.Fill, Font = new Font("Consolas", 10) };
    private readonly Button _subSend = new() { Text = "Send", Width = 60 };

    // ═══════════════ paint / theme ═══════════════
    static readonly Color ClChrome = Color.FromArgb(37, 39, 46);    // form, toolbars, headers
    static readonly Color ClPane   = Color.FromArgb(22, 23, 28);    // console panes
    static readonly Color ClInput  = Color.FromArgb(32, 34, 40);    // text inputs
    static readonly Color ClInk    = Color.Gainsboro;
    static readonly Color ClMuted  = Color.FromArgb(150, 158, 172);
    static readonly Color ClAccent = Color.FromArgb(64, 200, 214);  // cyan
    static readonly Color ClStop   = Color.FromArgb(226, 132, 96);  // amber (stop / danger)

    static Label Header(string t) => new()
    {
        Text = t, ForeColor = ClAccent, BackColor = ClChrome, Height = 22,
        Font = new Font("Consolas", 8.5f, FontStyle.Bold), TextAlign = ContentAlignment.MiddleLeft,
        Padding = new Padding(9, 0, 0, 0)
    };
    static Button Flat(Button b, bool primary = false)
    {
        b.FlatStyle = FlatStyle.Flat;
        b.FlatAppearance.BorderSize = 1;
        b.FlatAppearance.BorderColor = primary ? ClAccent : Color.FromArgb(72, 76, 86);
        b.BackColor = primary ? Color.FromArgb(28, 56, 62) : ClChrome;
        b.ForeColor = primary ? ClAccent : ClInk;
        b.Font = new Font("Segoe UI", 9);
        b.Height = 28; b.Margin = new Padding(3, 5, 3, 5);
        return b;
    }
    private void ApplyTheme()
    {
        BackColor = ClChrome; ForeColor = ClInk;
        foreach (var tb in new[] { _replIn, _askIn, _groupIn, _subReplIn }) { tb.BackColor = ClInput; tb.ForeColor = ClInk; tb.BorderStyle = BorderStyle.FixedSingle; }
        _hostCode.BackColor = ClPane; _hostCode.ForeColor = ClAccent; _hostCode.BorderStyle = BorderStyle.FixedSingle;
        foreach (var st in new[] { _status, _netStatus, _subStatus }) { st.BackColor = ClChrome; st.ForeColor = ClMuted; }
        _autoClear.ForeColor = ClMuted;
        _corpusOnly.ForeColor = ClMuted;
    }

    public MainForm()
    {
        Text = $"Prism Studio — {PrismSpec.Signature}";
        AutoScaleMode = AutoScaleMode.None;   // fixed hand-coded pixel layout → don't let WinForms rescale it (that collapsed the layout on high-DPI/scaled displays); Windows scales the whole window uniformly (see ApplicationHighDpiMode=DpiUnawareGdiScaled)
        Width = 1040; Height = 700; Font = new Font("Segoe UI", 9);
        StartPosition = FormStartPosition.CenterScreen;
        RefreshRepoData();   // every launch: copy the repo's shipped data into the persistent AppData folder (overwrite)
        _model = new StudioModel(_dataDir);
        _paramCount = _model.ParamCount;   // once
        ApplyTheme();   // apply the dark + cyan theme to the shared controls

        var tabs = new TabControl { Dock = DockStyle.Fill };
        var modelTab = new TabPage("Model") { Padding = new Padding(3), BackColor = ClChrome };
        modelTab.Controls.Add(BuildModelTab());
        var netTab = new TabPage("Network") { Padding = new Padding(3), BackColor = ClChrome };
        netTab.Controls.Add(BuildNetworkTab());
        var subTab = new TabPage("Sub-node") { Padding = new Padding(3), BackColor = ClChrome };
        subTab.Controls.Add(BuildSubnodeTab());
        tabs.TabPages.Add(modelTab);
        tabs.TabPages.Add(netTab);
        tabs.TabPages.Add(subTab);
        Controls.Add(tabs);

        if (_model.Load(_savePath)) Log($"[loaded] {_savePath}");
        else Log("[new model] no checkpoint — press Train (data ships in data/).");
        _chatFile = Path.Combine(_dataDir, "chat", "chat.txt");   // full transcript → reload + parsed into progressive (conversation → your reply) pairs
        try { if (File.Exists(_chatFile)) _transcript = File.ReadAllText(_chatFile); } catch { }
        if (_transcript.Length > 0) _transcript = ("\n" + _transcript).Replace("\nyou: ", "\nuser: ")[1..];   // migrate legacy "you:" → "user:" so the reloaded chat matches the new labels
        _transcript = StudioModel.CapRecent(_transcript, PrismSpec.Context);   // roll an old (uncapped) chat.txt down to the recent context window
        _replOut.AppendText(_transcript.Length > 0 ? RenderTranscript(_transcript) : "Prism chat — type and press Enter. It learns to reply like YOU, in context, and reloads.\r\n\r\n");
        _replOut.SelectionStart = _replOut.TextLength; _replOut.ScrollToCaret();
        ToggleAutoClear();   // auto-clear on by default
        // AUTO-JOIN the colony mesh on startup — no code to paste; the always-on anchor bootstraps us, then it's peer-to-peer
        Task.Run(() => { try { StartChatter(MqttRelay.DefaultRoom); } catch (Exception e) { NetLog("[chatter] auto-join failed: " + e.Message.Split('\n')[0]); } });
        UpdateStatus("idle");
    }

    private Control BuildModelTab()
    {
        var root = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 1, RowCount = 2, BackColor = ClChrome };
        root.RowStyles.Add(new RowStyle(SizeType.Absolute, 42));
        root.RowStyles.Add(new RowStyle(SizeType.Percent, 100));

        var bar = new FlowLayoutPanel { Dock = DockStyle.Fill, WrapContents = false, AutoScroll = true, BackColor = ClChrome };
        var openBtn = new Button { Text = "Open data folder", Width = 120 };
        var clearBtn = new Button { Text = "Clear log", Width = 70 };
        bar.Controls.AddRange(new Control[] { Flat(_trainBtn, true), Flat(openBtn), Flat(_saveBtn), Flat(_loadBtn), Flat(_resetBtn), Flat(clearBtn), _autoClear, _corpusOnly });

        // REPL (chat) on TOP, training log on BOTTOM — same horizontal split as the Network tab
        var split = new SplitContainer { Dock = DockStyle.Fill, Orientation = Orientation.Horizontal, SplitterDistance = 360 };

        // top pane = the chat REPL (left) beside a separate "ask a question" box (right), each with its OWN output
        var repl = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 2, RowCount = 1 };
        repl.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 62));
        repl.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 38));

        var chat = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 2, RowCount = 3 };
        chat.RowStyles.Add(new RowStyle(SizeType.Absolute, 22));
        chat.RowStyles.Add(new RowStyle(SizeType.Percent, 100));
        chat.RowStyles.Add(new RowStyle(SizeType.Absolute, 34));
        chat.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
        chat.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 64));
        var ch = Header("REPL — chat; continues your prompt in context, and learns your replies"); ch.Dock = DockStyle.Fill;
        chat.Controls.Add(ch, 0, 0); chat.SetColumnSpan(ch, 2);
        chat.Controls.Add(_replOut, 0, 1); chat.SetColumnSpan(_replOut, 2);
        chat.Controls.Add(_replIn, 0, 2); chat.Controls.Add(Flat(_sendBtn), 1, 2);

        var ask = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 2, RowCount = 3, Padding = new Padding(6, 0, 0, 0) };
        ask.RowStyles.Add(new RowStyle(SizeType.Absolute, 22));
        ask.RowStyles.Add(new RowStyle(SizeType.Percent, 100));
        ask.RowStyles.Add(new RowStyle(SizeType.Absolute, 34));
        ask.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
        ask.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 64));
        var ah = Header("ASK A QUESTION — no context, a straight-shot answer"); ah.Dock = DockStyle.Fill;
        ask.Controls.Add(ah, 0, 0); ask.SetColumnSpan(ah, 2);
        ask.Controls.Add(_askOut, 0, 1); ask.SetColumnSpan(_askOut, 2);
        ask.Controls.Add(_askIn, 0, 2); ask.Controls.Add(Flat(_askBtn), 1, 2);

        repl.Controls.Add(chat, 0, 0); repl.Controls.Add(ask, 1, 0);
        split.Panel1.Controls.Add(repl);

        var logHdr = Header("training log"); logHdr.Dock = DockStyle.Top;
        split.Panel2.Controls.Add(_log);
        split.Panel2.Controls.Add(logHdr);

        var outer = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 1, RowCount = 2, BackColor = ClChrome };
        outer.RowStyles.Add(new RowStyle(SizeType.Absolute, 24));
        outer.RowStyles.Add(new RowStyle(SizeType.Percent, 100));
        outer.Controls.Add(_status, 0, 0);
        outer.Controls.Add(split, 0, 1);

        root.Controls.Add(bar, 0, 0);
        root.Controls.Add(outer, 0, 1);

        _trainBtn.Click += (_, _) => ToggleTrain();
        openBtn.Click += (_, _) => { try { Directory.CreateDirectory(_dataDir); Process.Start(new ProcessStartInfo("explorer.exe", _dataDir) { UseShellExecute = true }); } catch { } };
        _saveBtn.Click += (_, _) => { _model.Save(_savePath); Log($"[saved] {_savePath}"); };
        _loadBtn.Click += (_, _) => { bool ok; ok = _model.Load(_savePath); Log(ok ? $"[loaded] {_savePath}" : "[load] no compatible checkpoint"); };
        _resetBtn.Click += (_, _) =>
        {
            if (MessageBox.Show(
                    $"Reset the model to a fresh {PrismSpec.Version} and delete the saved checkpoint?\n\n" +
                    "Your training data (text / pairs / chat) is kept — retrain to rebuild.\n" +
                    "This cannot be undone.",
                    "Reset model", MessageBoxButtons.OKCancel, MessageBoxIcon.Warning) != DialogResult.OK) return;
            _model.Reset(_savePath);
            Log($"[reset] fresh {PrismSpec.Signature} — checkpoint deleted; press Train to rebuild");
        };
        clearBtn.Click += (_, _) => _log.Clear();
        _autoClear.CheckedChanged += (_, _) => ToggleAutoClear();
        _corpusOnly.CheckedChanged += (_, _) => { _model.CorpusOnly = _corpusOnly.Checked; Log(_corpusOnly.Checked ? "[mix] corpus-only ON — next epoch trains data/text alone" : "[mix] corpus-only OFF — volume-weighted mix restored"); };
        _sendBtn.Click += (_, _) => Send();
        _replIn.KeyDown += (_, e) => { if (e.KeyCode == Keys.Enter) { e.SuppressKeyPress = true; Send(); } };
        _askBtn.Click += (_, _) => Ask();
        _askIn.KeyDown += (_, e) => { if (e.KeyCode == Keys.Enter) { e.SuppressKeyPress = true; Ask(); } };
        return root;
    }

    private void ToggleTrain()
    {
        if (_trainCts is null)
        {
            _trainCts = new CancellationTokenSource();
            _trainBtn.Text = "■ Stop"; _trainBtn.BackColor = Color.FromArgb(58, 34, 28); _trainBtn.ForeColor = ClStop; _trainBtn.FlatAppearance.BorderColor = ClStop;
            _saveBtn.Enabled = _loadBtn.Enabled = _resetBtn.Enabled = false;   // Save/Load/Reset contend with the training writer — off while training
            var ct = _trainCts.Token;
            _model.CorpusOnly = _corpusOnly.Checked;   // honour the toggle at train start (also updated live on CheckedChanged)
            if (!_model.Hosting) EnableShare();   // automatic — training always offers to the swarm
            _mainSaver = new System.Threading.Timer(_ => { try { _model.Save(_savePath); } catch { } }, null, 60000, 60000);   // save every 60s (open-ended run — don't only save on Stop)
            _bleedTimer = new System.Threading.Timer(_ =>
            {
                var ch = _chatter; if (ch == null) return;
                // Bleed is bounded + proximity-routed INSIDE the chatter: each push goes to the K CLOSEST peers (ping-ranked), so traffic stays O(1) network-wide
                if (_model.RandomPair(Random.Shared) is { } p) ch.SharePair(p.Prompt, p.Target);                       // bleed pairs
                if (_model.WeightSlice(Random.Shared, 1024) is { } ws) ch.ShareWeightSlice(ws.Start, ws.Vals);        // bleed a tiny weight slice → converge over time
                Task.Run(() =>   // bleed-chat: ask a peer what they'd say, then TRAIN on their reply (each model learns what the other expected it to say)
                {
                    var q = _model.RandomPrompt(Random.Shared);   // one of OUR chat contexts
                    if (q != null && ch.AskSwarm(q, 8000) is { } a && a.Continuation.Trim().Length > 0)
                    { _model.LearnFromPeer(q, a.Continuation.Trim()); NetLog("[bleed] ↔ distilled a peer's personality on our chat context"); }
                });
            }, null, 30000, 30000);
            Task.Run(() =>
            {
                try { _model.Train(int.MaxValue, 5.0, s => BeginInvoke(() => UpdateStatus(s)), s => BeginInvoke(() => Log(s)), ct); }   // open-ended — runs until Stop
                catch (Exception e) { BeginInvoke(() => Log("[train] " + e.Message)); }
                try { _model.Save(_savePath); } catch { }
                BeginInvoke(() => { _trainCts = null; _trainBtn.Text = "▶ Train"; _trainBtn.BackColor = Color.FromArgb(28, 56, 62); _trainBtn.ForeColor = ClAccent; _trainBtn.FlatAppearance.BorderColor = ClAccent; _trainBtn.Enabled = true; _saveBtn.Enabled = _loadBtn.Enabled = _resetBtn.Enabled = true; _bleedTimer?.Dispose(); _bleedTimer = null; _mainSaver?.Dispose(); _mainSaver = null; Log("[train] stopped · saved"); });
            });
        }
        else { _trainCts.Cancel(); _bleedTimer?.Dispose(); _bleedTimer = null; _mainSaver?.Dispose(); _mainSaver = null; _trainBtn.Text = "■ stopping…"; _trainBtn.Enabled = false; UpdateStatus("stopping…"); }
    }

    private void Send()
    {
        if (_replBusy) return;   // one turn at a time — no queueing up messages that all resolve at once
        var q = _replIn.Text.Trim();
        if (q.Length == 0) return;
        _replIn.Clear();
        _transcript += "user: " + q + "\n";   // human turn (a training TARGET). Serve primes "user: " for the reply — same slot as the group chat
        _replOut.AppendText("user: " + q + "\r\nprism: "); _replOut.SelectionStart = _replOut.TextLength; _replOut.ScrollToCaret();
        SetReplBusy(true);
        Task.Run(() =>
        {
            string reply; var viaSwarm = false;
            try
            {
                var local = _model.Serve(_transcript);   // uses the FULL context window (last 256 chars of the whole chat)
                if (_chatter is { } ch && local.Conf < EscalateMargin && ch.AskSwarm(_transcript, 8000) is { } s && s.Conf > local.Conf)
                    { reply = CleanReply(s.Continuation); viaSwarm = true; }
                else reply = CleanReply(local.Continuation);
            }
            catch (Exception e) { reply = "[error] " + e.Message; }
            _transcript = StudioModel.CapRecent(_transcript + "prism: " + reply + "\n", PrismSpec.Context);   // roll the saved context to the recent window (same as group chat) — never larger than the model's context
            SaveChat();
            BeginInvoke(() => { _replOut.AppendText(reply + (viaSwarm ? "  ⟵ swarm" : "") + "\r\n\r\n"); _replOut.SelectionStart = _replOut.TextLength; _replOut.ScrollToCaret(); SetReplBusy(false); });
        });
    }

    private void Ask()   // straight-shot: answer the question with NO conversation context; its own output, nothing saved or threaded into the chat
    {
        if (_replBusy) return;
        var q = _askIn.Text.Trim(); if (q.Length == 0) return;
        _askIn.Clear();
        _askOut.AppendText("Q: " + q + "\r\nA: "); _askOut.SelectionStart = _askOut.TextLength; _askOut.ScrollToCaret();
        SetReplBusy(true);
        Task.Run(() =>
        {
            string r;
            try { r = CleanReply(_model.Serve("user: " + q).Continuation); }   // just this prompt, no prior turns
            catch (Exception e) { r = "[error] " + e.Message; }
            BeginInvoke(() => { _askOut.AppendText(r + "\r\n\r\n"); _askOut.SelectionStart = _askOut.TextLength; _askOut.ScrollToCaret(); SetReplBusy(false); });
        });
    }

    // REPL busy state: lock input while the model/swarm is answering, with a live "responding…" indicator so it's clear it's still working (and you can't queue up messages)
    private void SetReplBusy(bool busy)
    {
        _replBusy = busy;
        _sendBtn.Enabled = _replIn.Enabled = _askBtn.Enabled = _askIn.Enabled = !busy;
        try { _replAnim?.Dispose(); } catch { } _replAnim = null;
        if (busy) { var n = 0; _replAnim = new System.Threading.Timer(_ => { try { BeginInvoke(() => _sendBtn.Text = new string('•', (n++ % 3) + 1)); } catch { } }, null, 0, 350); }
        else _sendBtn.Text = "Send";
    }

    private static string CleanReply(string s)
    {
        var nl = s.IndexOf('\n'); if (nl >= 0) s = s[..nl];   // one turn: stop at the first line break
        s = s.Trim();
        return s.Length == 0 ? "…" : s;
    }
    // reload: render the stored transcript with a BLANK LINE between exchanges (before each new "you:" turn) so it isn't a dense wall
    private static string RenderTranscript(string t) => t.Replace("\nuser: ", "\n\nuser: ").Replace("\n", "\r\n");
    private void SaveChat()
    {
        try { Directory.CreateDirectory(Path.GetDirectoryName(_chatFile)!); var tmp = _chatFile + ".tmp"; File.WriteAllText(tmp, _transcript); File.Move(tmp, _chatFile, true); } catch { }
    }

    private void EnableShare()
    {
        if (_model.Hosting) return;
        Task.Run(() =>
        {
            try { var code = _model.EnableHosting(m => NetLog("[net] " + m)); StartChatter(MqttRelay.DefaultRoom); BeginInvoke(() => { _hostCode.Text = code; Log("[share] training auto-shared to the swarm — code is on the Network tab (Copy + send)"); }); }   // chatter is ALWAYS the colony room (never the transient relay room) — no race with startup auto-join
            catch (Exception e) { BeginInvoke(() => Log("[share] not shared (" + e.Message.Split('\n')[0] + ") — training locally")); }
        });
    }

    private void StartChatter(string room)
    {
        if (_chatter != null) return;
        try { var ch = new SwarmChatter(room, _model.Serve, _model.AbsorbPair, (s, v) => Task.Run(() => _model.MergeWeightSlice(s, v, 0.05)), NetLog, onGroup: OnGroupMsg, groupServe: _model.GroupReply, absorbContext: _model.AbsorbChat, signature: PrismSpec.Signature); ch.Start(); _chatter = ch; NetLog("[chatter] peer chatter on — bleed (pairs + backprop-on-wrong + tiny weight-average) + networked REPL + group chat"); }
        catch (Exception e) { NetLog("[chatter] " + e.Message.Split('\n')[0]); }
    }
    private void StopChatter() { try { _chatter?.Stop(); } catch { } _chatter = null; }

    // ════════ Sub-node: a second in-process cell you start/stop from the UI ════════
    private void SubLog(string s) { _subLog.AppendText(s + "\r\n"); _subLog.SelectionStart = _subLog.TextLength; _subLog.ScrollToCaret(); }
    private void SubLogBg(string s) { try { BeginInvoke(() => SubLog(s)); } catch { } }
    private void SaveSubChat() { try { Directory.CreateDirectory(Path.GetDirectoryName(_subChatFile)!); var tmp = _subChatFile + ".tmp"; File.WriteAllText(tmp, _subTranscript); File.Move(tmp, _subChatFile, true); } catch { } }

    private Control BuildSubnodeTab()
    {
        var root = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 1, RowCount = 2, BackColor = ClChrome };
        root.RowStyles.Add(new RowStyle(SizeType.Absolute, 42));
        root.RowStyles.Add(new RowStyle(SizeType.Percent, 100));

        var bar = new FlowLayoutPanel { Dock = DockStyle.Fill, WrapContents = false, AutoScroll = true, BackColor = ClChrome };
        var clearBtn = new Button { Text = "Clear log", Width = 70 };
        bar.Controls.AddRange(new Control[] { Flat(_subBtn, true), Flat(clearBtn) });

        // sub-node REPL on TOP, training log on BOTTOM — same horizontal split as the other tabs
        var split = new SplitContainer { Dock = DockStyle.Fill, Orientation = Orientation.Horizontal, SplitterDistance = 360 };

        var repl = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 2, RowCount = 3 };
        repl.RowStyles.Add(new RowStyle(SizeType.Absolute, 22));
        repl.RowStyles.Add(new RowStyle(SizeType.Percent, 100));
        repl.RowStyles.Add(new RowStyle(SizeType.Absolute, 34));
        repl.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
        repl.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 64));
        var rh = Header("sub-node REPL — chat with this cell (a second, separate creature)"); rh.Dock = DockStyle.Fill;
        repl.Controls.Add(rh, 0, 0); repl.SetColumnSpan(rh, 2);
        repl.Controls.Add(_subReplOut, 0, 1); repl.SetColumnSpan(_subReplOut, 2);
        repl.Controls.Add(_subReplIn, 0, 2); repl.Controls.Add(Flat(_subSend), 1, 2);
        split.Panel1.Controls.Add(repl);

        var logHdr = Header("sub-node training log"); logHdr.Dock = DockStyle.Top;
        split.Panel2.Controls.Add(_subLog);
        split.Panel2.Controls.Add(logHdr);

        var outer = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 1, RowCount = 2, BackColor = ClChrome };
        outer.RowStyles.Add(new RowStyle(SizeType.Absolute, 24));
        outer.RowStyles.Add(new RowStyle(SizeType.Percent, 100));
        outer.Controls.Add(_subStatus, 0, 0);
        outer.Controls.Add(split, 0, 1);

        root.Controls.Add(bar, 0, 0);
        root.Controls.Add(outer, 0, 1);

        _subBtn.Click += (_, _) => ToggleSubnode();
        clearBtn.Click += (_, _) => _subLog.Clear();
        _subSend.Click += (_, _) => SubSend();
        _subReplIn.KeyDown += (_, e) => { if (e.KeyCode == Keys.Enter) { e.SuppressKeyPress = true; SubSend(); } };
        return root;
    }

    private void ToggleSubnode()
    {
        if (_subCts is null)   // ── START ──
        {
            _subCts = new CancellationTokenSource();
            _subBtn.Text = "■ Stop sub-node";
            var ct = _subCts.Token;
            _subStatus.Text = "sub-node: starting…"; SubLog("[sub] starting…");

            // EVERYTHING heavy (copy data, build the 1M model, deserialize the checkpoint) runs OFF the UI thread so the window never freezes.
            void Finish() => BeginInvoke(() =>
            {
                _subCts = null; _subBtn.Text = "▶ Start sub-node"; _subBtn.Enabled = true;
                _subStatus.Text = "sub-node: stopped · saved"; _subSaver?.Dispose(); _subSaver = null; _subBleedTimer?.Dispose(); _subBleedTimer = null;
                try { _subChatter?.Stop(); } catch { } _subChatter = null; SubLog("[sub] stopped · saved");
            });
            Task.Run(() =>
            {
                StudioModel m;
                try
                {
                    foreach (var lane in new[] { "pairs", "text" })   // curriculum → always refresh to the current data (its own copy, no file conflict with the main cell)
                    {
                        var s = Path.Combine(_dataDir, lane); var d = Path.Combine(_subDataDir, lane);
                        if (Directory.Exists(s)) { Directory.CreateDirectory(d); foreach (var f in Directory.EnumerateFiles(s)) File.Copy(f, Path.Combine(d, Path.GetFileName(f)), true); }
                    }
                    foreach (var lane in new[] { "chat", "group" })   // conversational context → INHERIT the main cell's whenever the sub-node's is EMPTY (so chat/group lanes have data and RandomPrompt has something to query peers with), then let the sub-node's own accumulate + diverge
                    {
                        var s = Path.Combine(_dataDir, lane); var d = Path.Combine(_subDataDir, lane);
                        var subEmpty = !Directory.Exists(d) || !Directory.EnumerateFiles(d).Any();
                        if (Directory.Exists(s) && Directory.EnumerateFiles(s).Any() && subEmpty) { Directory.CreateDirectory(d); foreach (var f in Directory.EnumerateFiles(s)) File.Copy(f, Path.Combine(d, Path.GetFileName(f)), true); }
                    }
                    m = new StudioModel(_subDataDir);
                    if (m.Load(_subSavePath)) SubLogBg($"[sub] resumed {_subSavePath}");
                    else if (m.Load(_savePath)) SubLogBg("[sub] seeded from the main checkpoint");
                    else SubLogBg("[sub] fresh model");
                    _subModel = m;
                    try { if (File.Exists(_subChatFile)) _subTranscript = StudioModel.CapRecent(File.ReadAllText(_subChatFile), PrismSpec.Context); } catch { }   // reload its own chat context (feeds the chat lane + bleed queries)
                    // Seed conversational context from the main cell's LIVE state when the sub's is empty — the file-copy above
                    // can miss (sub folder not empty / main hasn't flushed group.txt), so also seed IN-MEMORY so RandomPrompt has
                    // something to query peers with from the first tick. Both then accumulate their own network group chat + diverge.
                    try { if (_subTranscript.Length == 0 && _transcript.Length > 0) _subTranscript = StudioModel.CapRecent(_transcript, PrismSpec.Context); } catch { }
                    try { m.SeedGroup(_model.GroupContext); } catch { }
                    var render = _subTranscript.Length > 0 ? RenderTranscript(_subTranscript) : "";
                    BeginInvoke(() => { _subReplOut.Clear(); if (render.Length > 0) { _subReplOut.AppendText(render); _subReplOut.SelectionStart = _subReplOut.TextLength; _subReplOut.ScrollToCaret(); } });
                }
                catch (Exception e) { SubLogBg("[sub] start failed: " + e.Message.Split('\n')[0]); Finish(); return; }
                if (ct.IsCancellationRequested) { Finish(); return; }

                // always on the network, like the primary model — a full cell
                try
                {
                    _subChatter = new SwarmChatter(MqttRelay.DefaultRoom, m.Serve, m.AbsorbPair, (st, v) => Task.Run(() => m.MergeWeightSlice(st, v, 0.05)), SubLogBg, onGroup: (h, t) => m.AppendGroup(h, t), groupServe: m.GroupReply, absorbContext: m.AbsorbChat, signature: PrismSpec.Signature);
                    _subChatter.Start();
                    // proactive bleed (same as the main cell): share a pair + a weight slice, and QUERY a peer — which makes THAT peer backprop on our query context, and we distil its reply
                    _subBleedTimer = new System.Threading.Timer(_ =>
                    {
                        var ch = _subChatter; if (ch == null) return;
                        if (m.RandomPair(Random.Shared) is { } p) ch.SharePair(p.Prompt, p.Target);
                        if (m.WeightSlice(Random.Shared, 1024) is { } ws) ch.ShareWeightSlice(ws.Start, ws.Vals);
                        Task.Run(() =>
                        {
                            // [peer] lane needs BOTH a query context AND a peer that answers. Log every failure mode so it's
                            // never a silent black box — a passive anchor won't answer; an empty chat/group has nothing to ask.
                            var q = m.RandomPrompt(Random.Shared);
                            if (q == null) { SubLogBg("[sub] [peer] skipped — no chat/group context yet to query with (chat with the sub, or let it inherit the main cell's chat)"); return; }
                            var a = ch.AskSwarm(q, 8000);
                            if (a is not { } ans) { SubLogBg("[sub] [peer] no peer answered — needs an ACTIVE answering peer online (the anchor is passive; run the main cell or another node)"); return; }
                            var reply = ans.Continuation.Trim();
                            if (reply.Length == 0) { SubLogBg("[sub] [peer] a peer replied empty — stayed silent, nothing to distil"); return; }
                            m.LearnFromPeer(q, reply);
                            SubLogBg("[sub] ↔ [peer] distilled a peer's reply on our context — the peer lane will train on it");
                        });
                    }, null, 30000, 30000);
                    SubLogBg("[sub] on the network — full cell: answers, absorbs, shares, queries + distils, weight-averages");
                }
                catch (Exception e) { SubLogBg("[sub] network join failed: " + e.Message.Split('\n')[0]); }

                _subSaver = new System.Threading.Timer(_ => { try { m.Save(_subSavePath); } catch { } }, null, 60000, 60000);
                try { m.Train(int.MaxValue, 5.0, s => BeginInvoke(() => _subStatus.Text = "sub-node: " + s), SubLogBg, ct); }
                catch (Exception e) { SubLogBg("[sub] " + e.Message.Split('\n')[0]); }
                try { m.Save(_subSavePath); } catch { }
                Finish();
            });
        }
        else   // ── STOP ──
        {
            _subCts.Cancel(); _subBtn.Text = "■ stopping…"; _subBtn.Enabled = false;   // the train task saves + flips the button back when it lands
        }
    }

    private void SubSend()
    {
        if (_subBusy) return;
        var m = _subModel; if (m is null) { SubLog("[sub] start the sub-node first"); return; }
        var q = _subReplIn.Text.Trim(); if (q.Length == 0) return;
        _subReplIn.Clear();
        _subTranscript += "user: " + q + "\n";
        _subReplOut.AppendText("user: " + q + "\r\nprism: "); _subReplOut.SelectionStart = _subReplOut.TextLength; _subReplOut.ScrollToCaret();
        _subBusy = true; _subSend.Enabled = _subReplIn.Enabled = false;
        Task.Run(() =>
        {
            string reply;
            try { reply = CleanReply(m.Serve(_subTranscript).Continuation); }
            catch (Exception e) { reply = "[error] " + e.Message; }
            _subTranscript = StudioModel.CapRecent(_subTranscript + "prism: " + reply + "\n", PrismSpec.Context);
            SaveSubChat();   // persist → reloads next start + feeds the sub-node's chat training lane
            BeginInvoke(() => { _subReplOut.AppendText(reply + "\r\n\r\n"); _subReplOut.SelectionStart = _subReplOut.TextLength; _subReplOut.ScrollToCaret(); _subBusy = false; _subSend.Enabled = _subReplIn.Enabled = true; });
        });
    }

    private void UpdateStatus(string s)   // MUST be cheap — runs per batch on the UI thread. No file IO, no allocation.
    {
        var peers = _chatter?.PeerCount ?? 0;
        _status.Text = $"{_paramCount:N0} params{(peers > 0 ? $" · {peers} peer(s)" : "")} · {s}";
    }
    private void Log(string s) { _log.AppendText(s + "\r\n"); _log.SelectionStart = _log.TextLength; _log.ScrollToCaret(); }

    private void ToggleAutoClear()
    {
        _autoClearTimer?.Dispose(); _autoClearTimer = null;
        if (_autoClear.Checked)
            _autoClearTimer = new System.Threading.Timer(_ => BeginInvoke(() => { _log.Clear(); _netLog.Clear(); Log("[log] auto-cleared (hourly)"); }), null, TimeSpan.FromHours(1), TimeSpan.FromHours(1));
    }

    // ════════════════ Network tab ════════════════
    private Control BuildNetworkTab()
    {
        var root = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 1, RowCount = 3, BackColor = ClChrome };
        root.RowStyles.Add(new RowStyle(SizeType.Absolute, 46));
        root.RowStyles.Add(new RowStyle(SizeType.Absolute, 24));
        root.RowStyles.Add(new RowStyle(SizeType.Percent, 100));

        var bar = new FlowLayoutPanel { Dock = DockStyle.Fill, WrapContents = true, AutoScroll = true, BackColor = ClChrome };
        Label Cap(string t) => new() { Text = t, AutoSize = true, ForeColor = ClMuted, Padding = new Padding(12, 11, 0, 0) };
        var copyBtn = new Button { Text = "Copy", Width = 56 };
        var clearNetBtn = new Button { Text = "Clear log", Width = 70 };
        bar.Controls.AddRange(new Control[] {
            Cap("Auto-joined the colony ·  your share code:"), _hostCode, Flat(copyBtn), Flat(clearNetBtn) });

        // GROUP CHAT pane (top) over the network log (bottom)
        var group = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 2, RowCount = 3 };
        group.RowStyles.Add(new RowStyle(SizeType.Absolute, 22));
        group.RowStyles.Add(new RowStyle(SizeType.Percent, 100));
        group.RowStyles.Add(new RowStyle(SizeType.Absolute, 34));
        group.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
        group.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 68));
        var glbl = Header("GROUP CHAT — you + the network's models, round-robin (1 human turn → 1 AI reply). Trains only on human turns."); glbl.Dock = DockStyle.Fill;
        group.Controls.Add(glbl, 0, 0); group.SetColumnSpan(glbl, 2);
        group.Controls.Add(_groupOut, 0, 1); group.SetColumnSpan(_groupOut, 2);
        group.Controls.Add(_groupIn, 0, 2); group.Controls.Add(Flat(_groupSend), 1, 2);

        var split = new SplitContainer { Dock = DockStyle.Fill, Orientation = Orientation.Horizontal, SplitterDistance = 360 };
        split.Panel1.Controls.Add(group);
        var netHdr = Header("network log"); netHdr.Dock = DockStyle.Top;
        split.Panel2.Controls.Add(_netLog);
        split.Panel2.Controls.Add(netHdr);

        root.Controls.Add(bar, 0, 0);
        root.Controls.Add(_netStatus, 0, 1);
        root.Controls.Add(split, 0, 2);

        copyBtn.Click += (_, _) => { try { if (!_hostCode.Text.StartsWith("(")) { Clipboard.SetText(_hostCode.Text); NetLog("[net] share code copied"); } } catch { } };
        clearNetBtn.Click += (_, _) => _netLog.Clear();
        _groupSend.Click += (_, _) => GroupSend();
        _groupIn.KeyDown += (_, e) => { if (e.KeyCode == Keys.Enter) { e.SuppressKeyPress = true; GroupSend(); } };
        _netStatus.Text = "You auto-join the colony on startup. Train on the Model tab and it auto-shares. Pasting a code only needed to join a PRIVATE swarm instead of the colony.";
        return root;
    }

    // Group chat: local user turn → broadcast → round-robin ONE peer for the AI reply (or our own model if none) → broadcast it.
    private void GroupSend()
    {
        var text = _groupIn.Text.Trim(); if (text.Length == 0) return;
        _groupIn.Clear();
        _model.AppendGroup(true, text); AppendGroupLine("user", text);   // local HUMAN turn (a training target)
        var ch = _chatter; if (ch is null) { NetLog("[group] not connected to the mesh yet"); return; }
        ch.GroupBroadcast(true, text);
        Task.Run(async () => await ch.GroupAskNext(_model.GroupContext, reply =>
        {
            var r = (string.IsNullOrWhiteSpace(reply) ? _model.GroupReply(_model.GroupContext) : reply)?.Trim() ?? "";   // no peer answered → our own model replies
            if (r.Length == 0) return;
            _model.AppendGroup(false, r); ch.GroupBroadcast(false, r);   // AI turn = context only (never a training target)
            BeginInvoke(() => AppendGroupLine("prism", r));
        }));
    }

    // A group turn arrived from another node (broadcast) → record it for training + show it.
    private void OnGroupMsg(bool human, string text)
    {
        _model.AppendGroup(human, text);
        BeginInvoke(() => AppendGroupLine(human ? "user" : "prism", text));
    }

    private void AppendGroupLine(string who, string text)
    {
        _groupOut.AppendText($"{who}: {text}\r\n");
        _groupOut.SelectionStart = _groupOut.TextLength; _groupOut.ScrollToCaret();
    }

    private void NetLog(string s)
    {
        if (InvokeRequired) { BeginInvoke(() => NetLog(s)); return; }
        _netLog.AppendText(s + "\r\n"); _netLog.SelectionStart = _netLog.TextLength; _netLog.ScrollToCaret();
    }


    protected override void OnFormClosing(FormClosingEventArgs e)
    {
        _trainCts?.Cancel(); _subCts?.Cancel();
        _bleedTimer?.Dispose(); _mainSaver?.Dispose(); _autoClearTimer?.Dispose(); _subSaver?.Dispose(); _subBleedTimer?.Dispose(); StopChatter(); try { _subChatter?.Stop(); } catch { }
        SaveChat();
        try { _model.Save(_savePath); } catch { }
        try { _subModel?.Save(_subSavePath); } catch { }
        base.OnFormClosing(e);
    }
}
