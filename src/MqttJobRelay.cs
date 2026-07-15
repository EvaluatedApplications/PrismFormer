// Copyright (c) 2026 Dongyang Stephen Chen, trading as Evaluated Applications. All rights reserved.
// Source-available for study and verification; commercial use, redistribution, or derivatives require permission. See LICENSE.

using System.Collections.Concurrent;
using MQTTnet;
using MQTTnet.Client;

namespace PrismFormer;

/// <summary>
/// Minimal-traffic relay: the host sends a MANIFEST + the MODEL once, then every round ships only example POSITIONS
/// ("compute i..j") and receives gradients. Each worker reconstructs the actual training examples locally from the same
/// <see cref="IJobSource"/> it built from the manifest — so the data never crosses the wire (see <see cref="IJobSource"/>).
/// Same public MQTT broker as <see cref="MqttRelay"/>, so it works through any NAT with no port-forward / install.
/// </summary>
/// <summary>Connection-code helpers for the position relay (prefix PZJ1, distinct from the data relay's PZR1).</summary>
public static class MqttJob
{
    public static string MakeCode(string room) => Convert.ToBase64String(System.Text.Encoding.UTF8.GetBytes("PZJ1;" + room));
    public static string? ParseCode(string input) { try { var s = System.Text.Encoding.UTF8.GetString(Convert.FromBase64String((input ?? "").Trim())); return s.StartsWith("PZJ1;") ? s[5..] : null; } catch { return null; } }
}

public sealed class MqttJobHost : IJobTrainer
{
    public AlgFormer Model;
    readonly PrismTrainer _local;
    readonly IJobSource _source;
    readonly byte[] _manifest;
    IMqttClient? _client; string _room = ""; Action<string>? _onEvent;
    readonly object _lock = new();
    readonly List<string> _live = new(), _joining = new();
    readonly Dictionary<string, ConcurrentQueue<(byte tag, byte[] payload)>> _inbox = new();
    readonly Dictionary<string, double> _speed = new();
    double _hostSpeed, _lastLoss; long _lastRoundAt;
    readonly List<long> _buf = new();
    public int ChunkTarget = 4096;
    public int MinRoundIntervalMs = 400;

    public MqttJobHost(AlgFormer model, IJobSource source, byte[] manifest) { Model = model; _local = new PrismTrainer(model); _source = source; _manifest = manifest; }
    public int ActiveWorkers { get { lock (_lock) return _live.Count; } }
    public bool Hosting => _client?.IsConnected ?? false;

    public string StartRelay(Action<string>? onEvent = null, string? room = null)
    {
        _onEvent = onEvent; _room = room ?? MqttRelay.NewRoom();
        _client = new MqttFactory().CreateMqttClient();
        _client.ApplicationMessageReceivedAsync += e => { OnInbound(e.ApplicationMessage.PayloadSegment.ToArray()); return Task.CompletedTask; };
        _client.ConnectAsync(new MqttClientOptionsBuilder().WithTcpServer(MqttRelay.Broker, MqttRelay.Port).WithCleanSession().WithTimeout(TimeSpan.FromSeconds(10)).Build()).GetAwaiter().GetResult();
        _client.SubscribeAsync($"pz/{_room}/toHost").GetAwaiter().GetResult();
        onEvent?.Invoke($"relay room open via {MqttRelay.Broker} — workers reconstruct data locally, only gradients cross");
        return MqttJob.MakeCode(_room);
    }

    public void StopRelay()
    {
        try { lock (_lock) foreach (var id in _live) MqttRelay.Publish(_client!, $"pz/{_room}/toSlave/{id}", new[] { MqttRelay.DONE }); } catch { }
        try { _client?.DisconnectAsync().GetAwaiter().GetResult(); } catch { }
        _client = null; lock (_lock) { _live.Clear(); _joining.Clear(); _inbox.Clear(); _speed.Clear(); }
    }

    void OnInbound(byte[] m)
    {
        if (m.Length < 17) return;
        var id = Convert.ToHexString(m, 0, 16); var tag = m[16];
        if (tag == MqttRelay.HELLO) { lock (_lock) if (!_live.Contains(id) && !_joining.Contains(id)) { _joining.Add(id); _inbox[id] = new(); } _onEvent?.Invoke($"slave {id[..6]}… hello"); }
        else { ConcurrentQueue<(byte, byte[])>? q; lock (_lock) _inbox.TryGetValue(id, out q); q?.Enqueue((tag, m[17..])); }
    }

    void ToSlave(string id, byte tag, byte[] payload) { var msg = new byte[1 + payload.Length]; msg[0] = tag; Array.Copy(payload, 0, msg, 1, payload.Length); MqttRelay.Publish(_client!, $"pz/{_room}/toSlave/{id}", msg); }

    internal static byte[] PackPositions(IReadOnlyList<long> p) { using var ms = new MemoryStream(); var w = new BinaryWriter(ms); w.Write(p.Count); foreach (var x in p) w.Write(x); w.Flush(); return ms.ToArray(); }
    internal static long[] UnpackPositions(byte[] b) { using var r = new BinaryReader(new MemoryStream(b)); var n = r.ReadInt32(); var p = new long[n]; for (var i = 0; i < n; i++) p[i] = r.ReadInt64(); return p; }

    public double TrainPositions(IReadOnlyList<long> positions, double lr)
    {
        _buf.AddRange(positions);
        if (_buf.Count < ChunkTarget) return _lastLoss;   // coalesce into a big chunk -> one lean network round
        var chunk = _buf.ToArray(); _buf.Clear();
        return _lastLoss = DoRound(chunk, lr);
    }

    double DoRound(long[] pos, double lr)
    {
        if (pos.Length == 0) return _lastLoss;
        var wait = MinRoundIntervalMs - (Environment.TickCount64 - _lastRoundAt);
        if (_lastRoundAt > 0 && wait > 0) Thread.Sleep((int)wait);
        _lastRoundAt = Environment.TickCount64;

        List<string> workers, joiners;
        lock (_lock) { joiners = _joining.ToList(); _joining.Clear(); foreach (var id in joiners) _live.Add(id); workers = _live.ToList(); }
        foreach (var id in joiners) { ToSlave(id, MqttRelay.MANIFEST, _manifest); ToSlave(id, MqttRelay.MODEL, Model.Serialize()); _onEvent?.Invoke($"slave {id[..6]}… got manifest + model -> {ActiveWorkers} active"); }

        var n = workers.Count;
        var wt = new double[n + 1]; wt[0] = _hostSpeed > 0 ? _hostSpeed : 1.0;
        for (var i = 0; i < n; i++) wt[i + 1] = _speed.TryGetValue(workers[i], out var sp) && sp > 0 ? sp : 0.5 * wt[0];
        var wsum = wt.Sum(); var size = new int[n + 1]; var acc = 0;
        for (var i = 1; i <= n; i++) { size[i] = (int)Math.Round(pos.Length * wt[i] / wsum); acc += size[i]; }
        size[0] = Math.Max(0, pos.Length - acc);
        var off = new int[n + 2]; for (var i = 0; i <= n; i++) off[i + 1] = off[i] + Math.Min(size[i], pos.Length - off[i]);

        var sw = System.Diagnostics.Stopwatch.StartNew();
        for (var wi = 0; wi < n; wi++) ToSlave(workers[wi], MqttRelay.SHARD, PackPositions(pos[off[wi + 1]..off[wi + 2]]));   // ship POSITIONS, not data
        var sent = sw.Elapsed.TotalSeconds;
        var hostBatch = new List<(int[] Ctx, int Target)>(); for (var i = 0; i < off[1]; i++) hostBatch.Add(_source.GetExample(pos[i]));   // host reconstructs its own shard
        var merged = _local.AccumulateBatch(hostBatch, out var hostLoss);
        var hostSec = sw.Elapsed.TotalSeconds - sent;
        if (off[1] > 0 && hostSec > 1e-6) _hostSpeed = _hostSpeed <= 0 ? off[1] / hostSec : 0.6 * _hostSpeed + 0.4 * (off[1] / hostSec);

        var pending = new HashSet<string>(workers); var dead = new HashSet<string>();
        const double deadline = 20.0;
        while (pending.Count > 0 && sw.Elapsed.TotalSeconds - sent < deadline)
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

        var apply = DistNet.PackApply(lr, pos.Length, Model.SerializeGradient(merged));
        Model.Step(merged, lr, scale: pos.Length);
        foreach (var id in workers) if (!dead.Contains(id)) ToSlave(id, MqttRelay.APPLY, apply);
        if (dead.Count > 0) { lock (_lock) foreach (var id in dead) { _live.Remove(id); _speed.Remove(id); _inbox.Remove(id); } _onEvent?.Invoke($"{dead.Count} slave(s) dropped -> {ActiveWorkers} active"); }
        return hostLoss;
    }
}

/// <summary>A worker over the position-relay: build the local data source from the manifest, then every round turn a
/// list of positions into gradients (reconstructing the examples locally). The <paramref name="sourceFactory"/> maps a
/// manifest to the right <see cref="IJobSource"/> (BabyLM / Gym / synthetic).</summary>
public sealed class MqttJobWorker(Func<byte[], IJobSource> sourceFactory)
{
    public AlgFormer? Model { get; private set; }
    IJobSource? _source;
    IMqttClient? _client; string _room = ""; string _idHex = ""; byte[] _id = Array.Empty<byte>();
    readonly ManualResetEventSlim _done = new(false);
    readonly ConcurrentQueue<(byte tag, byte[] payload)> _in = new();
    Action<string>? _onEvent; int _contributed;

    public void JoinRoom(string code, Action<string>? onEvent = null, CancellationToken ct = default)
    {
        var room = MqttJob.ParseCode(code);
        if (room is null) { onEvent?.Invoke("not a position-relay code"); return; }
        _room = room; _onEvent = onEvent; _id = Guid.NewGuid().ToByteArray(); _idHex = Convert.ToHexString(_id);
        _client = new MqttFactory().CreateMqttClient();
        _client.ApplicationMessageReceivedAsync += e => { var p = e.ApplicationMessage.PayloadSegment.ToArray(); if (p.Length >= 1) _in.Enqueue((p[0], p[1..])); return Task.CompletedTask; };
        _client.ConnectAsync(new MqttClientOptionsBuilder().WithTcpServer(MqttRelay.Broker, MqttRelay.Port).WithCleanSession().WithTimeout(TimeSpan.FromSeconds(10)).Build()).GetAwaiter().GetResult();
        _client.SubscribeAsync($"pz/{_room}/toSlave/{_idHex}").GetAwaiter().GetResult();
        ToHost(MqttRelay.HELLO, Array.Empty<byte>());
        onEvent?.Invoke($"joined relay room via {MqttRelay.Broker} — waiting for the job manifest");
        using var reg = ct.Register(() => _done.Set());
        while (!_done.IsSet && !ct.IsCancellationRequested) { if (_in.TryDequeue(out var f)) Handle(f.tag, f.payload); else Thread.Sleep(3); }
        try { _client.DisconnectAsync().GetAwaiter().GetResult(); } catch { }
    }

    void ToHost(byte tag, byte[] payload) { var msg = new byte[16 + 1 + payload.Length]; Array.Copy(_id, msg, 16); msg[16] = tag; Array.Copy(payload, 0, msg, 17, payload.Length); MqttRelay.Publish(_client!, $"pz/{_room}/toHost", msg); }

    void Handle(byte tag, byte[] payload)
    {
        if (tag == MqttRelay.MANIFEST) { _source = sourceFactory(payload); _onEvent?.Invoke($"built local data source ({_source.Count:N0} examples) — no data will be sent"); }
        else if (tag == MqttRelay.MODEL) { Model = AlgFormer.Deserialize(payload); _onEvent?.Invoke($"synced host model ({Model.ParamCount:N0} params)"); }
        else if (tag == MqttRelay.SHARD && Model is not null && _source is not null)
        {
            var pos = MqttJobHost.UnpackPositions(payload); var g = Model.NewGrads();
            foreach (var i in pos) { var (c, t) = _source.GetExample(i); Model.Accumulate(c, t, g); }   // reconstruct locally
            ToHost(MqttRelay.GRAD, Model.SerializeGradient(g));
            if (++_contributed % 20 == 0) _onEvent?.Invoke($"contributed {_contributed} rounds");
        }
        else if (tag == MqttRelay.APPLY && Model is not null) { using var ms = new MemoryStream(payload); using var r = new BinaryReader(ms); var lr = r.ReadDouble(); var scale = r.ReadInt32(); var grad = r.ReadBytes(payload.Length - 12); Model.Step(Model.DeserializeGradient(grad), lr, scale); }
        else if (tag == MqttRelay.DONE) { _onEvent?.Invoke($"host finished — contributed {_contributed} rounds"); _done.Set(); }
    }
}
