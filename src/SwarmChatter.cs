// Copyright (c) 2026 Dongyang Stephen Chen, trading as Evaluated Applications. All rights reserved.
// Source-available for study and verification; commercial use, redistribution, or derivatives require permission. See LICENSE.

using System.Collections.Concurrent;
using System.Text;
using MQTTnet;
using MQTTnet.Client;

namespace PrismFormer;

/// <summary>
/// The peer-intelligence layer (see SWARM.md) — a PING-RANKED GOSSIP MESH, not a flat broadcast room, so it stays
/// O(1)-per-node as the swarm grows to thousands:
///   • DIRECTED inboxes — every node subscribes to its own <c>pzc/{room}/to/{id}</c> topic; the heavy traffic (pairs,
///     weight slices, queries + answers) is SENT to specific peers, never broadcast to all N.
///   • PROXIMITY — ping/pong measures per-peer RTT; bleeds and queries go to the K CLOSEST peers (ping-ranked), so a
///     node talks to its nearest neighbours first (cheapest hops), never the whole network.
///   • ROSTER GOSSIP — nodes periodically share a sample of the peer-ids they know with their closest neighbours, so
///     membership spreads TRANSITIVELY through the mesh instead of a global broadcast. A newly-heard id gets a directed
///     ping-probe: answer and you join the roster, stay silent and you never do (no immortal ghost peers).
///   • BOOTSTRAP — a fresh joiner announces ONCE with a lightweight HELLO on the shared room topic (the broker is the
///     only global rendezvous point); existing nodes reply directed with a roster sample. After that it's all directed.
/// One room code carries both this mesh and the gradient relay (MqttRelay). Serving a query runs off-thread so it never
/// stalls the broker pump. Validate over a real multi-machine network before relying on it.
/// </summary>
public sealed class SwarmChatter
{
    const byte HELLO = 1, PAIR = 2, QUERY = 3, ANSWER = 4, WSLICE = 5, PING = 6, PONG = 7, GOSSIP = 8, GROUPMSG = 9, GROUPASK = 10, GROUPREPLY = 11;
    readonly string _room;
    readonly byte[] _self = Guid.NewGuid().ToByteArray();   // 16 bytes
    readonly string _selfHex;
    readonly Func<string, (double Conf, string Continuation)> _serve;
    readonly Action<string, string> _onPair;
    readonly Action<int, double[]>? _onWeightSlice;
    readonly Action<string>? _log;
    readonly Action<bool, string>? _onGroup;             // a group-chat turn arrived (human?, text) → append + display
    readonly Func<string, string>? _groupServe;          // produce THIS node's group reply for a transcript (null = don't reply, e.g. anchor)
    readonly Action<string>? _absorbContext;             // we were QUERIED → learn from the human turns the query context carries
    readonly string _signature;                          // our model spec — advertised in HELLO; peers must match to join the mesh
    readonly ConcurrentDictionary<string, byte> _blockedLogged = new();   // incompatible peers we've already warned about (log once)
    readonly ConcurrentDictionary<int, TaskCompletionSource<string>> _groupPending = new();
    int _groupTurn;                                      // round-robin cursor over live peers
    IMqttClient? _c;
    Timer? _tick;
    volatile bool _stopping;
    int _answering;   // 0/1 gate: generate at most ONE answer at a time so queries can't pile up and pin a slow box
    int _qid, _tickCount;
    readonly ConcurrentDictionary<int, ConcurrentBag<(double Conf, string Cont)>> _answers = new();
    readonly ConcurrentDictionary<string, long> _peers = new();    // id -> last time we heard DIRECTLY from them (liveness)
    readonly ConcurrentDictionary<string, double> _rtt = new();    // id -> EWMA round-trip ms (proximity ranking)

    // ── mesh knobs: traffic per tick is independent of swarm size N ──
    public int AnswerQuorum = 3;   // a query goes to (and is answered by) only the K nearest peers
    public int BleedFanout = 3;    // a bleed / roster-gossip goes to only the K nearest peers
    public int PingSample = 6;     // RTT-probe this many random peers per tick (rolling proximity map)
    public int GossipSample = 8;   // share up to this many peer-ids per roster gossip
    public int PeerTtlMs = 300_000; // forget a peer we haven't heard from in this long → the roster can't grow unbounded
    public int MaxPeers = 4096;    // hard cap on remembered peers (evict the oldest beyond this, as a backstop)

    public SwarmChatter(string room, Func<string, (double Conf, string Continuation)> serve, Action<string, string> onPair, Action<int, double[]>? onWeightSlice = null, Action<string>? log = null,
                        Action<bool, string>? onGroup = null, Func<string, string>? groupServe = null, Action<string>? absorbContext = null, string? signature = null)
    { _room = room; _serve = serve; _onPair = onPair; _onWeightSlice = onWeightSlice; _log = log; _onGroup = onGroup; _groupServe = groupServe; _absorbContext = absorbContext; _signature = signature ?? ""; _selfHex = Convert.ToHexString(_self); }

    /// <summary>Does a peer's advertised spec match ours? Only exact-match peers may join the mesh (weight-slices are
    /// shape-specific). Empty = a legacy build that doesn't advertise → treated as incompatible.</summary>
    bool Compatible(string theirSig) => theirSig.Length > 0 && theirSig == _signature;

    public int PeerCount { get { var now = Environment.TickCount64; return _peers.Count(kv => now - kv.Value < 40_000); } }

    public void Start()
    {
        _c = new MqttFactory().CreateMqttClient();
        _c.ApplicationMessageReceivedAsync += e => { OnMsg(e.ApplicationMessage.PayloadSegment.ToArray()); return Task.CompletedTask; };
        // A plain MqttClient does NOT auto-reconnect: on any blip (keepalive miss when a slow box is busy generating,
        // broker restart, NAT timeout) it goes DEAF until process restart — the "anchor stopped seeing peers" bug.
        // Reconnect with backoff, re-subscribe, and re-announce so the node rejoins on its own.
        _c.DisconnectedAsync += async _ =>
        {
            if (_stopping) return;
            for (var attempt = 0; !_stopping; attempt++)
            {
                await Task.Delay(Math.Min(30_000, 2_000 * (attempt + 1)));
                try { await JoinAsync(); _log?.Invoke("[chatter] reconnected to the mesh"); return; }
                catch { }
            }
        };
        JoinAsync().GetAwaiter().GetResult();
        _tick = new Timer(_ => { try { Maintain(); } catch { } }, null, 4000, 8000);
        _log?.Invoke("[chatter] joined the mesh — discovering nearest peers");
    }

    // Connect (or re-connect) + subscribe (shared HELLO topic + personal directed inbox) + announce presence.
    async Task JoinAsync()
    {
        await _c!.ConnectAsync(MqttRelay.BuildOptions());
        await _c.SubscribeAsync($"pzc/{_room}");                 // shared: HELLO bootstrap only
        await _c.SubscribeAsync($"pzc/{_room}/to/{_selfHex}");   // personal inbox: all directed traffic
        Send(HELLO, Encoding.UTF8.GetBytes(_signature));         // announce (with our spec) so matching nodes bootstrap us
    }

    public void Stop() { _stopping = true; try { _tick?.Dispose(); _c?.DisconnectAsync().GetAwaiter().GetResult(); } catch { } _c = null; }

    // ── periodic maintenance: refresh proximity (ping) + spread membership (roster gossip) ──
    void Maintain()
    {
        // TRIM the roster: forget peers we haven't heard from in a while so memory stays bounded; hard-cap as a backstop
        var now = Environment.TickCount64;
        foreach (var kv in _peers) if (now - kv.Value > PeerTtlMs) { _peers.TryRemove(kv.Key, out _); _rtt.TryRemove(kv.Key, out _); }
        if (_peers.Count > MaxPeers)   // still over cap after the stale trim → drop the WORST-ping peers, keeping the nearest (unknown ping = treated as farthest)
            foreach (var id in _peers.Keys.OrderByDescending(k => _rtt.TryGetValue(k, out var r) ? r : double.MaxValue).Take(_peers.Count - MaxPeers))
            { _peers.TryRemove(id, out _); _rtt.TryRemove(id, out _); }
        var live = LivePeers();
        foreach (var id in Sample(live, PingSample)) SendTo(id, PING, BitConverter.GetBytes(Environment.TickCount64));   // RTT probes
        var roster = SampleIds(live, GossipSample);
        if (roster.Length > 0) foreach (var id in Closest(BleedFanout)) SendTo(id, GOSSIP, PackIds(roster));             // membership spreads to nearest neighbours
        if (++_tickCount % 6 == 0) Send(HELLO, Encoding.UTF8.GetBytes(_signature));   // rare shared keepalive (with our spec) so long-lived rooms stay discoverable
    }

    /// <summary>Broadcast a confident training pair — bled to the K NEAREST peers (they add it to gossip and learn it next epoch).</summary>
    public void SharePair(string prompt, string target)
    {
        var targets = Closest(BleedFanout); if (targets.Count == 0) return;
        using var ms = new MemoryStream(); var w = new BinaryWriter(ms);
        var p = Encoding.UTF8.GetBytes(prompt); var t = Encoding.UTF8.GetBytes(target);
        w.Write(p.Length); w.Write(p); w.Write(t.Length); w.Write(t); w.Flush();
        var payload = ms.ToArray();
        foreach (var id in targets) SendTo(id, PAIR, payload);
        _log?.Invoke($"[bleed] → shared a pair with {targets.Count} nearest peer(s)");
    }

    /// <summary>Bleed a tiny slice of weights (start index + values) to the K NEAREST peers — they elastic-average it, so models converge.</summary>
    public void ShareWeightSlice(int start, double[] vals)
    {
        var targets = Closest(BleedFanout); if (targets.Count == 0) return;
        using var ms = new MemoryStream(); var w = new BinaryWriter(ms);
        w.Write(start); w.Write(vals.Length); foreach (var v in vals) w.Write(v); w.Flush();
        var payload = ms.ToArray();
        foreach (var id in targets) SendTo(id, WSLICE, payload);
    }

    /// <summary>The networked REPL: ask the K NEAREST peers for a continuation, collect their answers briefly, return the
    /// most confident (or null if none answer).</summary>
    public (double Conf, string Continuation)? AskSwarm(string prompt, int timeoutMs = 8000)
    {
        if (_c is null) return null;
        var targets = Closest(AnswerQuorum); if (targets.Count == 0) return null;
        var qid = Interlocked.Increment(ref _qid);
        _answers[qid] = new();
        using (var ms = new MemoryStream()) { var w = new BinaryWriter(ms); w.Write(qid); var pb = Encoding.UTF8.GetBytes(prompt); w.Write(pb.Length); w.Write(pb); w.Flush(); var q = ms.ToArray(); foreach (var id in targets) SendTo(id, QUERY, q); }
        _log?.Invoke($"[query] → asked {targets.Count} nearest peer(s): \"{Trunc(prompt)}\"");
        var bag = _answers[qid];   // WAN, not LAN: give real answers time to make the round trip (broker→peer→serve→broker) before returning; only bail early once we're past the generous grace AND have answers, else wait the full ceiling
        const int grace = 3000;
        for (var waited = 0; waited < timeoutMs; waited += 100) { Thread.Sleep(100); if (!bag.IsEmpty && waited >= grace) break; }
        _answers.TryRemove(qid, out _);
        if (bag.IsEmpty) { _log?.Invoke("[query] → no peer answered in time"); return null; }
        var best = bag.OrderByDescending(a => a.Conf).First();
        _log?.Invoke($"[query] → {bag.Count} peer answer(s); took the most confident");
        return best;
    }

    // ════════ peer set / proximity helpers ════════
    List<string> LivePeers() { var now = Environment.TickCount64; return _peers.Where(kv => now - kv.Value < 40_000).Select(kv => kv.Key).ToList(); }

    /// <summary>The K nearest peers by measured RTT (unknown RTT ranks last), with a little exploration so we keep
    /// discovering closer neighbours.</summary>
    List<string> Closest(int k)
    {
        var live = LivePeers();
        if (live.Count <= k) return live;
        var pick = live.OrderBy(id => _rtt.TryGetValue(id, out var r) ? r : double.PositiveInfinity).Take(k).ToList();
        if (Random.Shared.NextDouble() < 0.25) pick[Random.Shared.Next(pick.Count)] = live[Random.Shared.Next(live.Count)];   // explore
        return pick;
    }

    static List<string> Sample(List<string> ids, int n) => ids.Count <= n ? ids : ids.OrderBy(_ => Random.Shared.Next()).Take(n).ToList();
    static string[] SampleIds(List<string> ids, int n) => Sample(ids, n).ToArray();
    byte[] PackIds(string[] ids) { using var ms = new MemoryStream(); var w = new BinaryWriter(ms); w.Write(ids.Length); foreach (var h in ids) w.Write(Convert.FromHexString(h)); w.Flush(); return ms.ToArray(); }

    static string Trunc(string s) => s.Length <= 40 ? s : s[..40] + "…";

    void OnMsg(byte[] m)
    {
        if (m.Length < 17 || m.AsSpan(0, 16).SequenceEqual(_self)) return;   // ignore malformed / our own
        var sender = Convert.ToHexString(m, 0, 16); var tag = m[16]; var payload = m[17..];
        if (tag == HELLO)   // a peer announced WITH ITS SPEC → verify before we ever add or bootstrap it
        {
            var theirSig = payload.Length > 0 ? Encoding.UTF8.GetString(payload) : "";
            if (!Compatible(theirSig))
            {
                if (_blockedLogged.Count > 4096) _blockedLogged.Clear();
                if (_blockedLogged.TryAdd(sender, 0)) _log?.Invoke($"[mesh] blocked a node on a different spec ({(theirSig.Length > 0 ? theirSig : "legacy build")}) — it must update to join");
                return;   // never bootstrap / add an incompatible peer — their weight-slices would corrupt ours
            }
            _peers[sender] = Environment.TickCount64;
            try { SendTo(sender, GOSSIP, PackIds(SampleIds(LivePeers(), GossipSample))); } catch { }   // bootstrap them with a roster sample
            return;
        }
        if (!_peers.ContainsKey(sender)) return;   // only ever interact with peers verified compatible via a HELLO
        _peers[sender] = Environment.TickCount64;   // known & compatible → refresh liveness
        try
        {
            switch (tag)
            {
                case GOSSIP:   // learn of peers we haven't met — PROBE each unknown one; it enters the roster only if it answers (PONG)
                {
                    using var r = new BinaryReader(new MemoryStream(payload));
                    var cnt = r.ReadInt32();
                    for (var i = 0; i < cnt && i < 4096; i++)
                    {
                        var id = Convert.ToHexString(r.ReadBytes(16));
                        if (id != _selfHex && !_peers.ContainsKey(id)) SendTo(id, PING, BitConverter.GetBytes(Environment.TickCount64));
                    }
                    break;
                }
                case PING:   // proximity probe → echo it back so the pinger can time the round trip
                {
                    var sent = payload.Length >= 8 ? BitConverter.ToInt64(payload, 0) : 0L;
                    SendTo(sender, PONG, BitConverter.GetBytes(sent));
                    break;
                }
                case PONG:
                {
                    var sent = payload.Length >= 8 ? BitConverter.ToInt64(payload, 0) : 0L;
                    var rtt = Math.Max(0, Environment.TickCount64 - sent);
                    _rtt.AddOrUpdate(sender, rtt, (_, old) => old * 0.7 + rtt * 0.3);
                    break;
                }
                case PAIR:
                {
                    using var r = new BinaryReader(new MemoryStream(payload));
                    var prompt = Encoding.UTF8.GetString(r.ReadBytes(r.ReadInt32()));
                    var target = Encoding.UTF8.GetString(r.ReadBytes(r.ReadInt32()));
                    _onPair(prompt, target);
                    _log?.Invoke($"[bleed] ← learned a pair from a peer: \"{Trunc(prompt)}\" → \"{Trunc(target)}\"");
                    break;
                }
                case QUERY:   // a nearest-peer asked us — serve off-thread and answer DIRECTED back to the asker
                {
                    using var r = new BinaryReader(new MemoryStream(payload));
                    var qid = r.ReadInt32(); var prompt = Encoding.UTF8.GetString(r.ReadBytes(r.ReadInt32()));
                    // We were ASKED → learn from the human turns the asker's context carries (they train us; we distil their reply on their side).
                    if (_absorbContext != null) { var p = prompt; Task.Run(() => { try { _absorbContext(p); } catch { } }); }
                    // SERIALIZE: generate at most one answer at a time. On a slow box a CPU-bound Serve starves the
                    // MQTT keepalive thread → the broker drops us; letting queries pile up guarantees it. If we're
                    // already answering, drop this one — the asker gets answers from other peers / retries.
                    if (Interlocked.CompareExchange(ref _answering, 1, 0) != 0) break;
                    Task.Run(() =>
                    {
                        try
                        {
                            var (conf, cont) = _serve(prompt);
                            if (string.IsNullOrEmpty(cont)) return;   // answer-disabled node / nothing to say → stay SILENT (don't put empty answers in the quorum)
                            _log?.Invoke($"[query] ← answered a peer: \"{Trunc(prompt)}\"");
                            using var ms = new MemoryStream(); var w = new BinaryWriter(ms);
                            w.Write(qid); w.Write(conf); var cb = Encoding.UTF8.GetBytes(cont); w.Write(cb.Length); w.Write(cb); w.Flush();
                            SendTo(sender, ANSWER, ms.ToArray());
                        }
                        catch { }
                        finally { Interlocked.Exchange(ref _answering, 0); }
                    });
                    break;
                }
                case ANSWER:
                {
                    using var r = new BinaryReader(new MemoryStream(payload));
                    var qid = r.ReadInt32(); var conf = r.ReadDouble(); var cont = Encoding.UTF8.GetString(r.ReadBytes(r.ReadInt32()));
                    if (_answers.TryGetValue(qid, out var bag)) bag.Add((conf, cont));
                    break;
                }
                case WSLICE:
                {
                    using var r = new BinaryReader(new MemoryStream(payload));
                    var start = r.ReadInt32(); var cnt = r.ReadInt32(); var vals = new double[cnt];
                    for (var i = 0; i < cnt; i++) vals[i] = r.ReadDouble();
                    _onWeightSlice?.Invoke(start, vals);   // merged off-thread by the caller
                    _log?.Invoke($"[bleed] ← averaged a weight slice from a peer");
                    break;
                }
                case GROUPMSG:   // a group-chat turn broadcast to the room → append to our local log + display
                {
                    using var r = new BinaryReader(new MemoryStream(payload));
                    var human = r.ReadBoolean(); var text = Encoding.UTF8.GetString(r.ReadBytes(r.ReadInt32()));
                    _onGroup?.Invoke(human, text);
                    break;
                }
                case GROUPASK:   // WE were picked (round-robin) to answer this group turn → reply in a human voice, directed back
                {
                    using var r = new BinaryReader(new MemoryStream(payload));
                    var qid = r.ReadInt32(); var transcript = Encoding.UTF8.GetString(r.ReadBytes(r.ReadInt32()));
                    if (_absorbContext != null) { var tr = transcript; Task.Run(() => { try { _absorbContext(tr); } catch { } }); }   // learn from the human turns in the group context (even if we decline to reply)
                    if (_groupServe is null)   // this node doesn't produce replies (e.g. the anchor) → decline FAST so the asker falls back now, not after a timeout
                    {
                        using var dm = new MemoryStream(); var dw = new BinaryWriter(dm); dw.Write(qid); dw.Write(0); dw.Flush();
                        SendTo(sender, GROUPREPLY, dm.ToArray()); break;
                    }
                    if (Interlocked.CompareExchange(ref _answering, 1, 0) != 0) break;    // serialize generation (shared with query answering)
                    Task.Run(() =>
                    {
                        try
                        {
                            var reply = _groupServe(transcript);
                            if (!string.IsNullOrWhiteSpace(reply))
                            {
                                using var ms = new MemoryStream(); var w = new BinaryWriter(ms);
                                w.Write(qid); var rb = Encoding.UTF8.GetBytes(reply); w.Write(rb.Length); w.Write(rb); w.Flush();
                                SendTo(sender, GROUPREPLY, ms.ToArray());
                            }
                        }
                        catch { }
                        finally { Interlocked.Exchange(ref _answering, 0); }
                    });
                    break;
                }
                case GROUPREPLY:   // the peer we asked answered → resolve the pending round-robin ask
                {
                    using var r = new BinaryReader(new MemoryStream(payload));
                    var qid = r.ReadInt32(); var reply = Encoding.UTF8.GetString(r.ReadBytes(r.ReadInt32()));
                    if (_groupPending.TryGetValue(qid, out var tcs)) tcs.TrySetResult(reply);
                    break;
                }
            }
        }
        catch { }
    }

    byte[] Frame(byte tag, byte[] payload) { var msg = new byte[17 + payload.Length]; Array.Copy(_self, 0, msg, 0, 16); msg[16] = tag; Array.Copy(payload, 0, msg, 17, payload.Length); return msg; }
    void Send(byte tag, byte[] payload) { if (_c is null) return; try { MqttRelay.Publish(_c, $"pzc/{_room}", Frame(tag, payload)); } catch { } }                       // shared room (bootstrap)
    void SendTo(string idHex, byte tag, byte[] payload) { if (_c is null) return; try { MqttRelay.Publish(_c, $"pzc/{_room}/to/{idHex}", Frame(tag, payload)); } catch { } }   // directed to one peer

    // ── GROUP CHAT ──────────────────────────────────────────────────────────────────────────────
    /// <summary>Broadcast a group-chat turn to the whole room (everyone appends it to their local log + displays it).</summary>
    public void GroupBroadcast(bool human, string text)
    {
        using var ms = new MemoryStream(); var w = new BinaryWriter(ms);
        w.Write(human); var tb = Encoding.UTF8.GetBytes(text); w.Write(tb.Length); w.Write(tb); w.Flush();
        Send(GROUPMSG, ms.ToArray());
    }

    /// <summary>Round-robin: pick the NEXT live peer (rotating) and ask it — and it alone — to produce one group reply,
    /// so replies don't all rush at once. <paramref name="onReply"/> gets the peer's text, or null if nobody's reachable
    /// / it timed out (the caller then falls back to its own model). One human turn → one AI reply.</summary>
    public async Task GroupAskNext(string transcript, Action<string?> onReply, int timeoutMs = 8000)
    {
        var peers = Closest(64);                                  // ping-ranked live peers
        if (peers.Count == 0) { onReply(null); return; }
        var target = peers[(Interlocked.Increment(ref _groupTurn) & int.MaxValue) % peers.Count];   // rotate → each node gets turns
        var qid = Interlocked.Increment(ref _qid);
        var tcs = new TaskCompletionSource<string>(TaskCreationOptions.RunContinuationsAsynchronously);
        _groupPending[qid] = tcs;
        try
        {
            using var ms = new MemoryStream(); var w = new BinaryWriter(ms);
            w.Write(qid); var tb = Encoding.UTF8.GetBytes(transcript); w.Write(tb.Length); w.Write(tb); w.Flush();
            SendTo(target, GROUPASK, ms.ToArray());
            var done = await Task.WhenAny(tcs.Task, Task.Delay(timeoutMs)).ConfigureAwait(false);
            onReply(done == tcs.Task ? tcs.Task.Result : null);
        }
        finally { _groupPending.TryRemove(qid, out _); }
    }
}
