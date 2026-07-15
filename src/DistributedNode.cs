// Copyright (c) 2026 Dongyang Stephen Chen, trading as Evaluated Applications. All rights reserved.
// Source-available for study and verification; commercial use, redistribution, or derivatives require permission. See LICENSE.

using System.Net;
using System.Net.Sockets;

namespace PrismFormer;

/// <summary>
/// Networked, elastic distributed training over a plain socket. A <see cref="DistributedHost"/> owns the job (model +
/// data) and manages a pool of <see cref="DistributedWorker"/>s ("slaves"): each round it delegates a data shard to
/// every node (sized by measured speed), sums their serialized gradients with its own, applies one
/// <see cref="AlgFormer.Step"/>, and broadcasts the merged update. Slaves register at any time (synced to the host's
/// current params at the round boundary) and are dropped gracefully if they disconnect or miss the round deadline.
/// The gradient merge is exact (see examples/DistributedTest); with stable membership + equal speeds it is bit-for-bit
/// identical to single-node. The host is an <see cref="IBatchTrainer"/>, so any training loop can host with no change.
/// </summary>
public static class DistNet
{
    internal const byte MODEL = 1, SHARD = 2, GRAD = 3, APPLY = 4, DONE = 5;

    internal static byte[] ReadExact(Stream s, int n) { var b = new byte[n]; var o = 0; while (o < n) { var r = s.Read(b, o, n - o); if (r <= 0) throw new EndOfStreamException(); o += r; } return b; }
    internal static void WriteFrame(Stream s, byte tag, byte[] payload) { s.Write(BitConverter.GetBytes(payload.Length + 1)); s.WriteByte(tag); s.Write(payload); s.Flush(); }
    internal static (byte tag, byte[] payload) ReadFrame(Stream s) { var len = BitConverter.ToInt32(ReadExact(s, 4)); var tag = ReadExact(s, 1)[0]; return (tag, ReadExact(s, len - 1)); }
    internal static byte[] PackBatch(IReadOnlyList<(int[] Ctx, int Target)> d) { using var ms = new MemoryStream(); var w = new BinaryWriter(ms); w.Write(d.Count); foreach (var (c, t) in d) { w.Write(c.Length); foreach (var x in c) w.Write(x); w.Write(t); } w.Flush(); return ms.ToArray(); }
    internal static List<(int[] Ctx, int Target)> UnpackBatch(byte[] p) { using var ms = new MemoryStream(p); using var r = new BinaryReader(ms); var n = r.ReadInt32(); var d = new List<(int[], int)>(n); for (var i = 0; i < n; i++) { var len = r.ReadInt32(); var c = new int[len]; for (var k = 0; k < len; k++) c[k] = r.ReadInt32(); d.Add((c, r.ReadInt32())); } return d; }
    internal static byte[] PackApply(double lr, int scale, byte[] grad) { using var ms = new MemoryStream(); var w = new BinaryWriter(ms); w.Write(lr); w.Write(scale); w.Write(grad); w.Flush(); return ms.ToArray(); }

    /// <summary>Discover this machine's public UDP endpoint via a free public STUN server (RFC 5389). Null if unreachable.</summary>
    public static string? PublicEndpoint(string server = "stun.l.google.com", int port = 19302)
    {
        try
        {
            using var udp = new UdpClient(); udp.Client.ReceiveTimeout = 3000;
            var req = new byte[20]; req[1] = 0x01; req[4] = 0x21; req[5] = 0x12; req[6] = 0xA4; req[7] = 0x42;
            var rng = new Random(); for (var i = 8; i < 20; i++) req[i] = (byte)rng.Next(256);
            udp.Send(req, req.Length, server, port);
            IPEndPoint? remote = null; var r = udp.Receive(ref remote);
            var pos = 20;
            while (pos + 4 <= r.Length)
            {
                int atype = (r[pos] << 8) | r[pos + 1], alen = (r[pos + 2] << 8) | r[pos + 3], v = pos + 4;
                if ((atype == 0x0020 || atype == 0x0001) && r[v + 1] == 0x01)
                {
                    var pt = (r[v + 2] << 8) | r[v + 3]; var ip = new byte[4]; Array.Copy(r, v + 4, ip, 0, 4);
                    if (atype == 0x0020) { pt ^= 0x2112; ip[0] ^= 0x21; ip[1] ^= 0x12; ip[2] ^= 0xA4; ip[3] ^= 0x42; }
                    return $"{ip[0]}.{ip[1]}.{ip[2]}.{ip[3]}:{pt}";
                }
                pos += 4 + alen + ((4 - alen % 4) % 4);
            }
            return null;
        }
        catch { return null; }
    }

    // ---- connection code: a compact base64 token the host shows and a slave pastes (carries its candidate endpoints) ----
    /// <summary>Build a base64 connection code carrying the host's reachable endpoints (public via STUN + LAN IPv4s) on
    /// <paramref name="port"/>. A slave pastes it; <see cref="DistributedWorker.JoinCode"/> tries each until one connects.</summary>
    public static string MakeConnectionCode(int port)
    {
        var eps = new List<string>();
        if (PublicEndpoint() is { } pub) eps.Add($"{pub.Split(':')[0]}:{port}");   // public IP + the TCP port (needs a port-forward/tunnel to be reachable)
        foreach (var ip in LocalIPv4()) eps.Add($"{ip}:{port}");
        return Convert.ToBase64String(System.Text.Encoding.UTF8.GetBytes("PZN1;" + string.Join("|", eps.Distinct())));
    }

    /// <summary>Parse a connection code (base64) OR a raw "host:port" into candidate endpoints.</summary>
    public static List<(string Host, int Port)> ParseConnectionCode(string input)
    {
        input = (input ?? "").Trim(); var list = new List<(string, int)>();
        try
        {
            var s = System.Text.Encoding.UTF8.GetString(Convert.FromBase64String(input));
            if (s.StartsWith("PZN1;"))
            {
                foreach (var ep in s[5..].Split('|', StringSplitOptions.RemoveEmptyEntries))
                { var p = ep.Split(':'); if (p.Length == 2 && int.TryParse(p[1], out var pt)) list.Add((p[0], pt)); }
                return list;
            }
        }
        catch { }
        var parts = input.Split(':');
        if (parts.Length == 2 && int.TryParse(parts[1], out var port)) list.Add((parts[0], port));
        return list;
    }

    static IEnumerable<string> LocalIPv4()
    {
        try { return Dns.GetHostAddresses(Dns.GetHostName()).Where(a => a.AddressFamily == AddressFamily.InterNetwork && !IPAddress.IsLoopback(a)).Select(a => a.ToString()).Distinct().ToList(); }
        catch { return Array.Empty<string>(); }
    }
}

/// <summary>A slave: connects to a host and contributes gradients on the data shards it is delegated, until the host
/// finishes or the connection drops (or it is cancelled).</summary>
public sealed class DistributedWorker
{
    public AlgFormer? Model { get; private set; }

    /// <summary>Join using a host CONNECTION CODE (base64) or a raw "host:port" — tries each endpoint the code carries
    /// until one connects, then contributes until the host finishes.</summary>
    public void JoinCode(string codeOrAddr, Action<string>? onEvent = null, CancellationToken ct = default)
    {
        var cands = DistNet.ParseConnectionCode(codeOrAddr);
        if (cands.Count == 0) { onEvent?.Invoke("invalid connection code / address"); return; }
        foreach (var (host, port) in cands)
        {
            if (ct.IsCancellationRequested) return;
            try { onEvent?.Invoke($"trying {host}:{port} …"); Join(host, port, onEvent, ct); return; }
            catch (Exception e) { onEvent?.Invoke($"  {host}:{port} unreachable ({e.Message.Split('\n')[0]})"); }
        }
        onEvent?.Invoke("could not reach the host on any address in the code");
    }

    public void Join(string host, int port, Action<string>? onEvent = null, CancellationToken ct = default, int connectTimeoutMs = 6000)
    {
        using var client = new TcpClient();
        using var reg = ct.Register(() => { try { client.Close(); } catch { } });
        if (!client.ConnectAsync(host, port).Wait(connectTimeoutMs)) throw new TimeoutException($"connect to {host}:{port} timed out");
        onEvent?.Invoke($"connected to {host}:{port} — contributing");
        var s = client.GetStream();
        var contributed = 0;
        while (!ct.IsCancellationRequested)
        {
            var (tag, p) = DistNet.ReadFrame(s);
            if (tag == DistNet.MODEL) { Model = AlgFormer.Deserialize(p); onEvent?.Invoke($"synced host model ({Model.ParamCount:N0} params)"); }
            else if (tag == DistNet.SHARD)
            {
                if (Model is null) continue;
                var shard = DistNet.UnpackBatch(p); var g = Model.NewGrads();
                foreach (var (c, t) in shard) Model.Accumulate(c, t, g);
                DistNet.WriteFrame(s, DistNet.GRAD, Model.SerializeGradient(g));
                if (++contributed % 20 == 0) onEvent?.Invoke($"contributed {contributed} rounds");
            }
            else if (tag == DistNet.APPLY && Model is not null) { using var ms = new MemoryStream(p); using var r = new BinaryReader(ms); var lr = r.ReadDouble(); var scale = r.ReadInt32(); var grad = r.ReadBytes(p.Length - 12); Model.Step(Model.DeserializeGradient(grad), lr, scale); }
            else if (tag == DistNet.DONE) { onEvent?.Invoke($"host finished — contributed {contributed} rounds"); break; }
        }
    }
}

/// <summary>The coordinator: an elastic worker pool + a one-batch training primitive. Open the pool with
/// <see cref="StartPool"/>; a training loop then calls <see cref="TrainBatch"/> per minibatch (delegated across the pool
/// plus the host's own local cores) and closes with <see cref="StopPool"/>. Because it implements
/// <see cref="IBatchTrainer"/>, any loop that trains through that interface becomes hostable with no other change.</summary>
public sealed class DistributedHost : IBatchTrainer
{
    public AlgFormer Model;
    readonly PrismTrainer _local;                               // the host's own shard, on local cores (EvalApp-tuned)
    readonly object _lock = new();
    readonly List<(TcpClient C, NetworkStream S)> _live = new();
    readonly List<(TcpClient C, NetworkStream S)> _joining = new();
    readonly Dictionary<NetworkStream, double> _speed = new();   // per-slave throughput (examples/sec) EMA -> load balancing
    double _hostSpeed;
    TcpListener? _listener; Thread? _accept; CancellationTokenSource? _poolCts; Action<string>? _onEvent;

    public DistributedHost(AlgFormer model) { Model = model; _local = new PrismTrainer(model); }
    public int ActiveWorkers { get { lock (_lock) return _live.Count; } }
    public bool Hosting => _listener is not null;

    /// <summary>Open the pool on <paramref name="port"/> — slaves may register at any time.</summary>
    public void StartPool(int port, Action<string>? onEvent = null)
    {
        _onEvent = onEvent; _poolCts = new CancellationTokenSource(); var ct = _poolCts.Token;
        _listener = new TcpListener(IPAddress.Any, port); _listener.Start();
        onEvent?.Invoke($"pool open on :{port} — public endpoint {DistNet.PublicEndpoint() ?? "n/a"}");
        _accept = new Thread(() => { try { while (!ct.IsCancellationRequested) { var c = _listener.AcceptTcpClient(); var s = c.GetStream(); s.ReadTimeout = s.WriteTimeout = 8000; lock (_lock) _joining.Add((c, s)); } } catch { } }) { IsBackground = true };
        _accept.Start();
    }

    /// <summary>Close the pool: tell slaves the job is done and drop the listener.</summary>
    public void StopPool()
    {
        _poolCts?.Cancel();
        try { _listener?.Stop(); } catch { }
        lock (_lock) { foreach (var w in _live) { try { DistNet.WriteFrame(w.S, DistNet.DONE, Array.Empty<byte>()); w.C.Close(); } catch { } } _live.Clear(); foreach (var w in _joining) { try { w.C.Close(); } catch { } } _joining.Clear(); _speed.Clear(); }
        _listener = null;
    }

    /// <summary>One distributed round: size a contiguous shard of <paramref name="batch"/> for each node by measured
    /// speed, compute the host's shard on local cores, merge all gradients, apply one Step, broadcast the update.
    /// Slow slaves get less work; a slave past the round deadline is dropped. Returns the host-shard mean loss.</summary>
    public double TrainBatch(IReadOnlyList<(int[] Ctx, int Target)> batch, double lr, CancellationToken ct = default)
    {
        if (batch.Count == 0) return 0;
        List<(TcpClient C, NetworkStream S)> workers;
        lock (_lock)
        {
            foreach (var j in _joining) try { DistNet.WriteFrame(j.S, DistNet.MODEL, Model.Serialize()); _live.Add(j); _onEvent?.Invoke($"slave joined -> {_live.Count} active"); } catch { try { j.C.Close(); } catch { } }
            _joining.Clear(); workers = _live.ToList();
        }
        var n = workers.Count;
        var dead = new HashSet<(TcpClient C, NetworkStream S)>();

        // AUTO-BALANCE: shard sized by each node's measured throughput, so a slow slave does less and finishes about when
        // the fast ones do — the slowest never paces the round. New slaves start conservative (half the host's rate).
        var wt = new double[n + 1]; wt[0] = _hostSpeed > 0 ? _hostSpeed : 1.0;
        for (var i = 0; i < n; i++) wt[i + 1] = _speed.TryGetValue(workers[i].S, out var sp) && sp > 0 ? sp : 0.5 * wt[0];
        var wsum = wt.Sum(); var size = new int[n + 1]; var acc = 0;
        for (var i = 1; i <= n; i++) { size[i] = (int)Math.Round(batch.Count * wt[i] / wsum); acc += size[i]; }
        size[0] = Math.Max(0, batch.Count - acc);
        var off = new int[n + 2]; for (var i = 0; i <= n; i++) off[i + 1] = off[i] + Math.Min(size[i], batch.Count - off[i]);

        var sw = System.Diagnostics.Stopwatch.StartNew();
        for (var wi = 0; wi < n; wi++) { try { DistNet.WriteFrame(workers[wi].S, DistNet.SHARD, DistNet.PackBatch(Slice(batch, off[wi + 1], off[wi + 2]))); } catch { dead.Add(workers[wi]); } }
        var sent = sw.Elapsed.TotalSeconds;
        var merged = _local.AccumulateBatch(Slice(batch, 0, off[1]), out var hostLoss);   // host shard on local cores, grad only
        var hostSec = sw.Elapsed.TotalSeconds - sent;
        if (off[1] > 0 && hostSec > 1e-6) _hostSpeed = _hostSpeed <= 0 ? off[1] / hostSec : 0.6 * _hostSpeed + 0.4 * (off[1] / hostSec);

        var pending = new List<int>(); for (var i = 0; i < n; i++) if (!dead.Contains(workers[i])) pending.Add(i);
        const double deadline = 8.0;
        while (pending.Count > 0 && sw.Elapsed.TotalSeconds - sent < deadline)
        {
            for (var pi = pending.Count - 1; pi >= 0; pi--)
            {
                var wi = pending[pi]; var wc = workers[wi];
                bool ready; try { ready = wc.S.DataAvailable; } catch { dead.Add(wc); pending.RemoveAt(pi); continue; }
                if (!ready) continue;
                try { var (t, p) = DistNet.ReadFrame(wc.S); if (t == DistNet.GRAD) merged.Add(Model.DeserializeGradient(p)); } catch { dead.Add(wc); pending.RemoveAt(pi); continue; }
                var lat = sw.Elapsed.TotalSeconds - sent; var shardN = off[wi + 2] - off[wi + 1];
                if (shardN > 0 && lat > 1e-6) { var s2 = shardN / lat; _speed[wc.S] = _speed.TryGetValue(wc.S, out var old) && old > 0 ? 0.6 * old + 0.4 * s2 : s2; }
                pending.RemoveAt(pi);
            }
            if (pending.Count > 0) Thread.Sleep(1);
        }
        foreach (var wi in pending) dead.Add(workers[wi]);   // missed the round deadline (hung) -> drop; can re-join re-synced

        var gradBytes = Model.SerializeGradient(merged);
        Model.Step(merged, lr, scale: batch.Count);
        var apply = DistNet.PackApply(lr, batch.Count, gradBytes);
        foreach (var wc in workers) { if (dead.Contains(wc)) continue; try { DistNet.WriteFrame(wc.S, DistNet.APPLY, apply); } catch { dead.Add(wc); } }
        if (dead.Count > 0) { lock (_lock) foreach (var d in dead) { _live.Remove(d); _speed.Remove(d.S); try { d.C.Close(); } catch { } } _onEvent?.Invoke($"{dead.Count} slave(s) dropped/timed-out -> {_live.Count} active"); }
        return hostLoss;
    }

    static List<(int[] Ctx, int Target)> Slice(IReadOnlyList<(int[] Ctx, int Target)> b, int from, int to) { var l = new List<(int[], int)>(Math.Max(0, to - from)); for (var i = from; i < to; i++) l.Add(b[i]); return l; }

    /// <summary>Convenience: run a whole self-contained job (demo / self-test) — open pool, wait for
    /// <paramref name="minStart"/> slaves, train, close.</summary>
    public void Run(IReadOnlyList<(int[] Ctx, int Target)> data, int epochs, int batch, double lr, int port, int minStart,
        Action<string>? onEvent = null, Action<(int round, double loss, int workers)>? onRound = null, CancellationToken ct = default)
    {
        StartPool(port, onEvent);
        using var reg = ct.Register(StopPool);
        while (!ct.IsCancellationRequested) { int have; lock (_lock) have = _live.Count + _joining.Count; if (have >= minStart) break; Thread.Sleep(20); }
        var round = 0;
        for (var ep = 0; ep < epochs && !ct.IsCancellationRequested; ep++)
            for (var start = 0; start < data.Count && !ct.IsCancellationRequested; start += batch)
            {
                var loss = TrainBatch(Slice(data, start, Math.Min(start + batch, data.Count)), lr);
                round++; onRound?.Invoke((round, loss, ActiveWorkers));
            }
        StopPool();
        onEvent?.Invoke("host job finished");
    }
}
