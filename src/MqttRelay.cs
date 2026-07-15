// Copyright (c) 2026 Dongyang Stephen Chen, trading as Evaluated Applications. All rights reserved.
// Source-available for study and verification; commercial use, redistribution, or derivatives require permission. See LICENSE.

using System.Collections.Concurrent;
using MQTTnet;
using MQTTnet.Client;
using MQTTnet.Protocol;

namespace PrismFormer;

/// <summary>
/// Zero-setup RELAY transport for distributed training: the host and every slave dial OUT to a free public MQTT broker
/// and exchange the training protocol through it. Because both sides only make outbound connections, it works through
/// any home NAT with no port-forward and nothing to install — the host shows a short room CODE, a slave pastes it, done.
/// Honest tradeoff: all traffic passes through the public broker, so it's great for small models / demos but slow for
/// big-gradient training (direct P2P / Tailscale is faster). No message chunking yet, so keep models modest.
/// </summary>
public static class MqttRelay
{
    // Broker endpoint + auth are configurable so you can point Prism at YOUR OWN broker (a $5/mo VPS running Mosquitto/EMQX
    // with a big max_packet_size + no flood limits makes the 8 MB-per-round gradient relay actually viable — the free public
    // broker can't). Set these env vars on every node; defaults keep the zero-setup public broker for the mesh/demo.
    //   PRISM_BROKER, PRISM_BROKER_PORT, PRISM_BROKER_USER, PRISM_BROKER_PASS, PRISM_BROKER_TLS=1
    public const string DefaultBroker = "79.72.78.90";      // the always-on anchor's broker (Oracle Always-Free box) → every client auto-connects here with zero config (override with PRISM_BROKER)
    public const string DefaultRoomName = "prism-colony";   // the one well-known colony everyone auto-joins (override with PRISM_ROOM)
    public static string Broker => Environment.GetEnvironmentVariable("PRISM_BROKER") is { Length: > 0 } b ? b : DefaultBroker;
    public static int Port => int.TryParse(Environment.GetEnvironmentVariable("PRISM_BROKER_PORT"), out var p) && p > 0 ? p : 1883;
    /// <summary>The well-known colony room every client auto-joins with no code (override with PRISM_ROOM).</summary>
    public static string DefaultRoom => Environment.GetEnvironmentVariable("PRISM_ROOM") is { Length: > 0 } r ? r : DefaultRoomName;

    /// <summary>Client options for the configured broker, including optional credentials + TLS. Used by every connection
    /// (relay host, relay worker, and the chatter mesh) so one set of env vars repoints the whole stack.</summary>
    internal static MqttClientOptions BuildOptions()
    {
        var b = new MqttClientOptionsBuilder().WithTcpServer(Broker, Port).WithCleanSession().WithTimeout(TimeSpan.FromSeconds(10));
        var user = Environment.GetEnvironmentVariable("PRISM_BROKER_USER");
        if (!string.IsNullOrEmpty(user)) b = b.WithCredentials(user, Environment.GetEnvironmentVariable("PRISM_BROKER_PASS") ?? "");
        if (Environment.GetEnvironmentVariable("PRISM_BROKER_TLS") == "1") b = b.WithTlsOptions(o => { });
        return b.Build();
    }
    internal const byte MODEL = 1, SHARD = 2, GRAD = 3, APPLY = 4, DONE = 5, HELLO = 6, MANIFEST = 7, CHUNK = 8;
    internal const int MaxChunk = 180_000;   // max payload bytes per broker message: a model / gradient is ~8 MB, far over any public broker's single-message cap, so big sends are split into this-many-byte CHUNK frames and reassembled on the far side

    /// <summary>Frame one CHUNK body: [msgId 4][idx 2][count 2][origTag 1][data slice]. Reassembled by <see cref="ChunkReassembler"/>.</summary>
    internal static byte[] ChunkFrame(int msgId, int idx, int count, byte origTag, byte[] src, int off, int len)
    {
        var b = new byte[9 + len];
        BitConverter.GetBytes(msgId).CopyTo(b, 0);
        BitConverter.GetBytes((ushort)idx).CopyTo(b, 4);
        BitConverter.GetBytes((ushort)count).CopyTo(b, 6);
        b[8] = origTag;
        Array.Copy(src, off, b, 9, len);
        return b;
    }

    public static string NewRoom() => Guid.NewGuid().ToString("N")[..10];
    /// <summary>A pasteable relay code (base64) carrying the room id. Distinct from the direct-IP connection code.</summary>
    public static string MakeRoomCode(string room) => Convert.ToBase64String(System.Text.Encoding.UTF8.GetBytes("PZR1;" + room));
    /// <summary>Room id from a relay code, or null if it isn't one.</summary>
    public static string? ParseRoomCode(string input)
    {
        try { var s = System.Text.Encoding.UTF8.GetString(Convert.FromBase64String((input ?? "").Trim())); return s.StartsWith("PZR1;") ? s[5..] : null; }
        catch { return null; }
    }

    internal static IMqttClient Connect()
    {
        var c = new MqttFactory().CreateMqttClient();
        c.ConnectAsync(BuildOptions()).GetAwaiter().GetResult();
        return c;
    }
    internal static void Publish(IMqttClient c, string topic, byte[] payload) =>
        c.PublishAsync(new MqttApplicationMessageBuilder().WithTopic(topic).WithPayload(payload).WithQualityOfServiceLevel(MqttQualityOfServiceLevel.AtLeastOnce).Build()).GetAwaiter().GetResult();
}

/// <summary>Reassembles a large logical message (model / gradient) from CHUNK frames — see <see cref="MqttRelay.CHUNK"/>.
/// Dedupes repeated chunks (QoS-1 may redeliver) and yields the full payload + original tag once every chunk has arrived.
/// Single-threaded use per stream (host: one per slave on the broker-callback thread; worker: one on its loop thread).</summary>
sealed class ChunkReassembler
{
    sealed class Asm { public byte Tag; public byte[][] Parts = null!; public bool[] Got = null!; public int Have; public int Count; }
    readonly Dictionary<int, Asm> _m = new();
    public bool Feed(byte[] cp, out byte tag, out byte[] full)
    {
        tag = 0; full = Array.Empty<byte>();
        if (cp.Length < 9) return false;
        var msgId = BitConverter.ToInt32(cp, 0);
        int idx = BitConverter.ToUInt16(cp, 4), count = BitConverter.ToUInt16(cp, 6);
        if (count == 0 || idx >= count) return false;
        if (!_m.TryGetValue(msgId, out var a)) { a = new Asm { Tag = cp[8], Parts = new byte[count][], Got = new bool[count], Count = count }; _m[msgId] = a; }
        if (!a.Got[idx]) { a.Parts[idx] = cp[9..]; a.Got[idx] = true; a.Have++; }
        if (a.Have != a.Count) return false;
        _m.Remove(msgId);
        var total = 0; foreach (var p in a.Parts) total += p.Length;
        full = new byte[total]; var o = 0; foreach (var p in a.Parts) { Array.Copy(p, 0, full, o, p.Length); o += p.Length; }
        tag = a.Tag; return true;
    }
}

/// <summary>Host over the relay: manages a slave pool keyed by id and delegates shards, exactly like
/// <see cref="DistributedHost"/> but transported through the public broker. Implements <see cref="IBatchTrainer"/>.</summary>
public sealed class MqttRelayHost : IBatchTrainer
{
    public AlgFormer Model;
    readonly PrismTrainer _local;
    IMqttClient? _client;
    string _room = "";
    Action<string>? _onEvent;
    readonly object _lock = new();
    readonly List<string> _live = new();
    readonly List<string> _joining = new();
    readonly Dictionary<string, ConcurrentQueue<(byte tag, byte[] payload)>> _inbox = new();
    readonly Dictionary<string, double> _speed = new();
    double _hostSpeed;
    readonly List<(int[] Ctx, int Target)> _buf = new();
    double _lastLoss; long _lastRoundAt;
    int _seq;                                                      // monotonic id for chunked sends
    readonly Dictionary<string, ChunkReassembler> _rasm = new();   // per-slave reassembly of chunked GRAD
    public int ChunkTarget = 4096;        // coalesce small batches into big network rounds -> far fewer broker messages ("bigger chunks, less chatter")
    public int MinRoundIntervalMs = 500;  // and never publish rounds faster than this -> stays under public-broker flood limits

    public MqttRelayHost(AlgFormer model) { Model = model; _local = new PrismTrainer(model); }
    public int ActiveWorkers { get { lock (_lock) return _live.Count; } }
    public bool Hosting => _client?.IsConnected ?? false;

    /// <summary>Open a relay room and return the pasteable code. Slaves that paste it join the pool.</summary>
    public string StartRelay(Action<string>? onEvent = null, string? room = null)
    {
        _onEvent = onEvent; _room = room ?? MqttRelay.NewRoom();
        _client = new MqttFactory().CreateMqttClient();
        _client.ApplicationMessageReceivedAsync += e => { OnInbound(e.ApplicationMessage.PayloadSegment.ToArray()); return Task.CompletedTask; };
        _client.ConnectAsync(MqttRelay.BuildOptions()).GetAwaiter().GetResult();
        _client.SubscribeAsync($"pz/{_room}/toHost").GetAwaiter().GetResult();
        onEvent?.Invoke($"relay room open via {MqttRelay.Broker} — slaves can paste the code to join");
        return MqttRelay.MakeRoomCode(_room);
    }

    public void StopRelay()
    {
        try { lock (_lock) foreach (var id in _live) MqttRelay.Publish(_client!, $"pz/{_room}/toSlave/{id}", new[] { MqttRelay.DONE }); } catch { }
        try { _client?.DisconnectAsync().GetAwaiter().GetResult(); } catch { }
        _client = null; lock (_lock) { _live.Clear(); _joining.Clear(); _inbox.Clear(); _speed.Clear(); }
    }

    void OnInbound(byte[] m)   // [16 slaveId][1 tag][payload]
    {
        if (m.Length < 17) return;
        var id = Convert.ToHexString(m, 0, 16); var tag = m[16];
        if (tag == MqttRelay.HELLO) { lock (_lock) if (!_live.Contains(id) && !_joining.Contains(id)) { _joining.Add(id); _inbox[id] = new(); } _onEvent?.Invoke($"slave {id[..6]}… hello"); }
        else if (tag == MqttRelay.CHUNK)   // reassemble a chunked GRAD, then hand the whole message to the round loop
        {
            ChunkReassembler ra; lock (_lock) { if (!_rasm.TryGetValue(id, out ra!)) { ra = new(); _rasm[id] = ra; } }
            if (ra.Feed(m[17..], out var otag, out var full)) { ConcurrentQueue<(byte, byte[])>? q; lock (_lock) _inbox.TryGetValue(id, out q); q?.Enqueue((otag, full)); }
        }
        else { ConcurrentQueue<(byte, byte[])>? q; lock (_lock) _inbox.TryGetValue(id, out q); q?.Enqueue((tag, m[17..])); }
    }

    void ToSlave(string id, byte tag, byte[] payload)
    {
        var topic = $"pz/{_room}/toSlave/{id}";
        if (payload.Length <= MqttRelay.MaxChunk) { MqttRelay.Publish(_client!, topic, SlaveFrame(tag, payload)); return; }
        var msgId = Interlocked.Increment(ref _seq);   // split big MODEL / SHARD / APPLY sends across chunks
        var n = (payload.Length + MqttRelay.MaxChunk - 1) / MqttRelay.MaxChunk;
        for (var i = 0; i < n; i++)
        {
            var off = i * MqttRelay.MaxChunk; var len = Math.Min(MqttRelay.MaxChunk, payload.Length - off);
            MqttRelay.Publish(_client!, topic, SlaveFrame(MqttRelay.CHUNK, MqttRelay.ChunkFrame(msgId, i, n, tag, payload, off, len)));
        }
    }
    static byte[] SlaveFrame(byte tag, byte[] payload) { var m = new byte[1 + payload.Length]; m[0] = tag; Array.Copy(payload, 0, m, 1, payload.Length); return m; }

    public double TrainBatch(IReadOnlyList<(int[] Ctx, int Target)> batch, double lr, CancellationToken ct = default)
    {
        int peers; lock (_lock) peers = _live.Count + _joining.Count;
        if (peers == 0)   // NO peers → train THIS batch locally & immediately, cancellable. An empty room never buffers or blocks.
        {
            if (_buf.Count > 0) { var flush = _buf.ToList(); _buf.Clear(); _local.TrainBatch(flush, lr, ct); }   // drain any buffer left from when peers were present
            return _lastLoss = _local.TrainBatch(batch, lr, ct);
        }
        _buf.AddRange(batch);
        if (_buf.Count < ChunkTarget) return _lastLoss;   // peers present → amortize into ONE network round
        var chunk = _buf.ToList(); _buf.Clear();
        return _lastLoss = DoRound(chunk, lr, ct);
    }

    double DoRound(IReadOnlyList<(int[] Ctx, int Target)> batch, double lr, CancellationToken ct)
    {
        if (batch.Count == 0) return 0;
        var wait = MinRoundIntervalMs - (Environment.TickCount64 - _lastRoundAt);   // pace the public broker
        if (_lastRoundAt > 0 && wait > 0) Thread.Sleep((int)wait);
        _lastRoundAt = Environment.TickCount64;
        List<string> workers, joiners;
        lock (_lock) { joiners = _joining.ToList(); _joining.Clear(); foreach (var id in joiners) _live.Add(id); workers = _live.ToList(); }
        foreach (var id in joiners) { ToSlave(id, MqttRelay.MODEL, Model.Serialize()); _onEvent?.Invoke($"slave {id[..6]}… synced -> {ActiveWorkers} active"); }

        var n = workers.Count;
        var wt = new double[n + 1]; wt[0] = _hostSpeed > 0 ? _hostSpeed : 1.0;
        for (var i = 0; i < n; i++) wt[i + 1] = _speed.TryGetValue(workers[i], out var sp) && sp > 0 ? sp : 0.5 * wt[0];
        var wsum = wt.Sum(); var size = new int[n + 1]; var acc = 0;
        for (var i = 1; i <= n; i++) { size[i] = (int)Math.Round(batch.Count * wt[i] / wsum); acc += size[i]; }
        size[0] = Math.Max(0, batch.Count - acc);
        var off = new int[n + 2]; for (var i = 0; i <= n; i++) off[i + 1] = off[i] + Math.Min(size[i], batch.Count - off[i]);

        var sw = System.Diagnostics.Stopwatch.StartNew();
        for (var wi = 0; wi < n; wi++) ToSlave(workers[wi], MqttRelay.SHARD, DistNet.PackBatch(Slice(batch, off[wi + 1], off[wi + 2])));
        var sent = sw.Elapsed.TotalSeconds;
        var merged = _local.AccumulateBatch(Slice(batch, 0, off[1]), out var hostLoss);
        var hostSec = sw.Elapsed.TotalSeconds - sent;
        if (off[1] > 0 && hostSec > 1e-6) _hostSpeed = _hostSpeed <= 0 ? off[1] / hostSec : 0.6 * _hostSpeed + 0.4 * (off[1] / hostSec);

        var pending = new HashSet<string>(workers); var dead = new HashSet<string>();
        const double deadline = 20.0;   // relay latency is higher than a socket, so a more generous round deadline
        while (pending.Count > 0 && sw.Elapsed.TotalSeconds - sent < deadline && !ct.IsCancellationRequested)
        {
            foreach (var id in pending.ToList())
            {
                ConcurrentQueue<(byte tag, byte[] payload)>? q; lock (_lock) _inbox.TryGetValue(id, out q);
                if (q is null) { dead.Add(id); pending.Remove(id); continue; }
                if (!q.TryDequeue(out var f)) continue;
                if (f.tag == MqttRelay.GRAD) merged.Add(Model.DeserializeGradient(f.payload));
                var lat = sw.Elapsed.TotalSeconds - sent; var idx = workers.IndexOf(id); var shardN = off[idx + 2] - off[idx + 1];
                if (shardN > 0 && lat > 1e-6) { var s2 = shardN / lat; _speed[id] = _speed.TryGetValue(id, out var old) && old > 0 ? 0.6 * old + 0.4 * s2 : s2; }
                pending.Remove(id);
            }
            if (pending.Count > 0) Thread.Sleep(3);
        }
        foreach (var id in pending) dead.Add(id);

        var apply = DistNet.PackApply(lr, batch.Count, Model.SerializeGradient(merged));
        Model.Step(merged, lr, scale: batch.Count);
        foreach (var id in workers) if (!dead.Contains(id)) ToSlave(id, MqttRelay.APPLY, apply);
        if (dead.Count > 0) { lock (_lock) foreach (var id in dead) { _live.Remove(id); _speed.Remove(id); _inbox.Remove(id); _rasm.Remove(id); } _onEvent?.Invoke($"{dead.Count} slave(s) dropped -> {ActiveWorkers} active"); }
        return hostLoss;
    }

    static List<(int[] Ctx, int Target)> Slice(IReadOnlyList<(int[] Ctx, int Target)> b, int from, int to) { var l = new List<(int[], int)>(Math.Max(0, to - from)); for (var i = from; i < to; i++) l.Add(b[i]); return l; }
}

/// <summary>A slave over the relay: paste the room code, contribute gradients through the broker.</summary>
public sealed class MqttRelayWorker
{
    public AlgFormer? Model { get; private set; }
    IMqttClient? _client; string _room = ""; string _idHex = ""; byte[] _id = Array.Empty<byte>();
    readonly ManualResetEventSlim _done = new(false);
    readonly ConcurrentQueue<(byte tag, byte[] payload)> _in = new();   // received messages, processed off the MQTT callback
    Action<string>? _onEvent; int _contributed; int _seq; readonly ChunkReassembler _ra = new();

    public void JoinRoom(string code, Action<string>? onEvent = null, CancellationToken ct = default)
    {
        var room = MqttRelay.ParseRoomCode(code);
        if (room is null) { onEvent?.Invoke("not a relay code"); return; }
        _room = room; _onEvent = onEvent; _id = Guid.NewGuid().ToByteArray(); _idHex = Convert.ToHexString(_id);
        _client = new MqttFactory().CreateMqttClient();
        // the callback ONLY enqueues — computing the gradient + publishing here would stall MQTTnet's receive pump
        _client.ApplicationMessageReceivedAsync += e => { var p = e.ApplicationMessage.PayloadSegment.ToArray(); if (p.Length >= 1) _in.Enqueue((p[0], p[1..])); return Task.CompletedTask; };
        _client.ConnectAsync(MqttRelay.BuildOptions()).GetAwaiter().GetResult();
        _client.SubscribeAsync($"pz/{_room}/toSlave/{_idHex}").GetAwaiter().GetResult();
        ToHost(MqttRelay.HELLO, Array.Empty<byte>());
        onEvent?.Invoke($"joined relay room via {MqttRelay.Broker} — waiting for work");
        using var reg = ct.Register(() => _done.Set());
        var lastHello = Environment.TickCount64;
        while (!_done.IsSet && !ct.IsCancellationRequested)   // process on this (background) thread
        {
            if (_in.TryDequeue(out var f)) Handle(f.tag, f.payload);
            else Thread.Sleep(3);
            // Re-announce until the host has synced us: our first HELLO may have been missed if the host subscribed after
            // we joined (clean session = no retained messages). Idempotent host-side; stops once we receive the model.
            if (Model is null && Environment.TickCount64 - lastHello > 4000)
            { try { ToHost(MqttRelay.HELLO, Array.Empty<byte>()); } catch { } lastHello = Environment.TickCount64; }
        }
        try { _client.DisconnectAsync().GetAwaiter().GetResult(); } catch { }
    }

    void ToHost(byte tag, byte[] payload)
    {
        var topic = $"pz/{_room}/toHost";
        if (payload.Length <= MqttRelay.MaxChunk) { MqttRelay.Publish(_client!, topic, HostFrame(tag, payload)); return; }
        var msgId = ++_seq;   // split the big GRAD send across chunks
        var n = (payload.Length + MqttRelay.MaxChunk - 1) / MqttRelay.MaxChunk;
        for (var i = 0; i < n; i++)
        {
            var off = i * MqttRelay.MaxChunk; var len = Math.Min(MqttRelay.MaxChunk, payload.Length - off);
            MqttRelay.Publish(_client!, topic, HostFrame(MqttRelay.CHUNK, MqttRelay.ChunkFrame(msgId, i, n, tag, payload, off, len)));
        }
    }
    byte[] HostFrame(byte tag, byte[] payload) { var m = new byte[16 + 1 + payload.Length]; Array.Copy(_id, m, 16); m[16] = tag; Array.Copy(payload, 0, m, 17, payload.Length); return m; }

    void Handle(byte tag, byte[] payload)
    {
        if (tag == MqttRelay.CHUNK) { if (_ra.Feed(payload, out var otag, out var full)) Handle(otag, full); return; }   // reassemble chunked MODEL / SHARD / APPLY
        if (tag == MqttRelay.MODEL) { Model = AlgFormer.Deserialize(payload); _onEvent?.Invoke($"synced host model ({Model.ParamCount:N0} params)"); }
        else if (tag == MqttRelay.SHARD && Model is not null) { var shard = DistNet.UnpackBatch(payload); var g = Model.NewGrads(); foreach (var (c, t) in shard) Model.Accumulate(c, t, g); ToHost(MqttRelay.GRAD, Model.SerializeGradient(g)); if (++_contributed % 20 == 0) _onEvent?.Invoke($"contributed {_contributed} rounds"); }
        else if (tag == MqttRelay.APPLY && Model is not null) { using var ms = new MemoryStream(payload); using var r = new BinaryReader(ms); var lr = r.ReadDouble(); var scale = r.ReadInt32(); var grad = r.ReadBytes(payload.Length - 12); Model.Step(Model.DeserializeGradient(grad), lr, scale); }
        else if (tag == MqttRelay.DONE) { _onEvent?.Invoke($"host finished — contributed {_contributed} rounds"); _done.Set(); }
    }
}
