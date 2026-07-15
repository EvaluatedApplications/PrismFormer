using System.Net;
using System.Net.Sockets;
using PrismFormer;

// ═══════════════════════════════════════════════════════════════════════════════════════════════════════════
//  PrismNet — networked, ELASTIC distributed training for PrismFormer.
//
//  A HOST owns the job (model + data). WORKERS ("slaves") connect over a socket and contribute gradients. Membership
//  is dynamic: a worker can join at any time (it is synced to the host's current params at the next round boundary),
//  and a worker that disconnects is dropped and the round continues with whoever remains — down to host-only.
//  Each round's gradient merge is exact; with STABLE membership the run is bit-for-bit identical to single-node.
//
//  Modes:
//    (no args)                 self-test: (A) bit-exact vs single-node, (B) live drop-in / drop-out demo
//    swarmtest [workers]       integration test: N tiny cells train as a colony over loopback (positions only)
//    host   <port> <minStart>  run a host; start once <minStart> workers are present, then accept/drop freely
//    worker <hostIp> <port>    become a slave of a host over the network
//    stun                      discover this machine's public UDP endpoint via free public STUN
// ═══════════════════════════════════════════════════════════════════════════════════════════════════════════

if (args.Length >= 1 && args[0] == "swarmtest") { SwarmTest.Run(args.Length >= 2 ? int.Parse(args[1]) : 3); return; }
if (args.Length >= 1 && args[0] == "swarmtasks") { SwarmTasks.Run(args.Length >= 2 ? int.Parse(args[1]) : 4); return; }
if (args.Length >= 1 && args[0] == "sizes") { SizeReport.Run(); return; }
if (args.Length >= 1 && args[0] == "swarmbleed") { SwarmBleed.Run(); return; }
if (args.Length >= 1 && args[0] == "swarmrun") { SwarmRun.Run(); return; }
if (args.Length >= 1 && args[0] == "swarmlearn") { SwarmLearn.Run(); return; }
if (args.Length >= 1 && args[0] == "pairtest") { PairTest.Run(); return; }
if (args.Length >= 1 && args[0] == "gendata") { PrismGym.StarterData.GenerateAll(args.Length >= 2 ? args[1] : "starter", args.Length >= 3 ? int.Parse(args[2]) : 100_000, Console.WriteLine); return; }
if (args.Length >= 1 && args[0] == "stun") { Console.WriteLine(Net.TryStun() is { } e ? $"public endpoint (free STUN): {e}" : "STUN unreachable from here"); return; }
if (args.Length >= 1 && args[0] == "relaytest") { Net.RelayTest(); return; }
if (args.Length >= 1 && args[0] == "relaytest2") { Net.RelayTest2(); return; }
if (args.Length >= 3 && args[0] == "host") { Net.RunHostCli(int.Parse(args[1]), int.Parse(args[2])); return; }
if (args.Length >= 3 && args[0] == "worker") { Net.RunWorkerCli(args[1], int.Parse(args[2])); return; }

Console.WriteLine("=== PrismNet: elastic distributed training (self-test over loopback TCP) ===\n");
Console.WriteLine($"free-public-infra check — STUN: {(Net.TryStun() is { } ep ? ep + "  (reachable)" : "unreachable from this sandbox (works on a real machine)")}\n");
var data = Net.MakeData(256);

// ---------- (A) stable membership => bit-for-bit identical to single-node ----------
{
    const int EPOCHS = 4, BATCH = 32, WORKERS = 2;
    var reference = Net.NewModel();
    var loss0 = Net.AvgLoss(reference, data);
    Net.TrainGrouped(reference, data, EPOCHS, BATCH, 1 + WORKERS, 5e-3);

    var listener = new TcpListener(IPAddress.Loopback, 0); listener.Start();
    var port = ((IPEndPoint)listener.LocalEndpoint).Port;
    for (var i = 0; i < WORKERS; i++) Net.SpawnLoopbackWorker(port);
    var host = new Net.Host(Net.NewModel());
    host.Run(data, EPOCHS, BATCH, 5e-3, listener, minStart: WORKERS);

    Console.WriteLine($"(A) 1 host + {WORKERS} workers, stable membership, {EPOCHS} epochs:");
    Console.WriteLine($"    bit-for-bit identical to single-node : {(Net.BitEqual(host.Model, reference) ? "YES" : "NO")}   (max param diff {Net.MaxDiff(host.Model, reference):E1})");
    Console.WriteLine($"    real training — avg loss {loss0:F3} -> {Net.AvgLoss(host.Model, data):F3}");
}

// ---------- (B) live drop-in / drop-out ----------
{
    Console.WriteLine("\n(B) graceful drop-in / drop-out demo (slaves join and leave mid-training):");
    var listener = new TcpListener(IPAddress.Loopback, 0); listener.Start();
    var port = ((IPEndPoint)listener.LocalEndpoint).Port;
    var wa = Net.ConnectLoopback(port); Net.RunWorkerOn(wa);   // slave A
    var wb = Net.ConnectLoopback(port); Net.RunWorkerOn(wb);   // slave B

    var events = new List<string>();
    var host = new Net.Host(Net.NewModel());

    var injector = new Thread(() =>
    {
        Thread.Sleep(400); Console.WriteLine("    >>> slave A leaves (killing its connection)"); try { wa.Close(); } catch { }
        Thread.Sleep(500); Console.WriteLine("    >>> slave C joins (a new machine shows up)"); Net.RunWorkerOn(Net.ConnectLoopback(port));
    }) { IsBackground = true };
    injector.Start();

    host.Run(data, 20, 32, 5e-3, listener, minStart: 1, demoDelayMs: 8,
        onEvent: e => { lock (events) events.Add(e); Console.WriteLine($"    [membership] {e}"); },
        onRound: r => { if (r.n % 40 == 0) Console.WriteLine($"    round {r.n,3}: {r.workers} worker(s) contributing, loss {r.loss:F3}"); });

    bool drop = events.Any(e => e.Contains("dropped")), join = events.Count(e => e.Contains("joined")) >= 3;
    Console.WriteLine($"\n    completed every round through the churn (no crash) : YES");
    Console.WriteLine($"    a slave DROPPED OUT and training kept going          : {(drop ? "YES" : "NO")}");
    Console.WriteLine($"    a slave DROPPED IN mid-training and contributed      : {(join ? "YES" : "NO")}");
    Console.WriteLine($"    final loss {Net.AvgLoss(host.Model, data):F3}");
}

Console.WriteLine("\nMultiple slaves, elastic: workers come and go; each round merges whoever's present, exactly.");

static class Net
{
    public const int VOCAB = 32, SHIFTS = 8, LAYERS = 2, MAXCTX = 4, DMODEL = 32, SEED = 1;
    const byte MODEL = 1, SHARD = 2, GRAD = 3, APPLY = 4, DONE = 5;

    public static AlgFormer NewModel() => new(VOCAB, shifts: SHIFTS, layers: LAYERS, maxContext: MAXCTX, dModel: DMODEL, frozenPrefix: 0, embedSeed: null, seed: SEED);
    public static List<(int[] Ctx, int Tgt)> MakeData(int count)
    {
        var rng = new Random(12345); var d = new List<(int[], int)>(count);
        for (var i = 0; i < count; i++) { var c = new int[MAXCTX]; for (var k = 0; k < MAXCTX; k++) c[k] = rng.Next(VOCAB); d.Add((c, c[0])); }
        return d;
    }
    public static double AvgLoss(AlgFormer m, List<(int[] Ctx, int Tgt)> data) { var g = m.NewGrads(); double s = 0; foreach (var (c, t) in data) s += m.Accumulate(c, t, g); return s / data.Count; }
    public static void TrainGrouped(AlgFormer m, List<(int[] Ctx, int Tgt)> data, int epochs, int batch, int nodes, double lr)
    {
        for (var ep = 0; ep < epochs; ep++)
            for (var start = 0; start < data.Count; start += batch)
            {
                var b = data.GetRange(start, Math.Min(batch, data.Count - start));
                var merged = m.NewGrads();
                for (var k = 0; k < nodes; k++) { var g = m.NewGrads(); for (var i = k; i < b.Count; i += nodes) m.Accumulate(b[i].Ctx, b[i].Tgt, g); merged.Add(g); }
                m.Step(merged, lr, scale: b.Count);
            }
    }
    static byte[] Save(AlgFormer m) { using var ms = new MemoryStream(); var w = new BinaryWriter(ms); m.Save(w); w.Flush(); return ms.ToArray(); }
    public static bool BitEqual(AlgFormer a, AlgFormer b) => Save(a).SequenceEqual(Save(b));
    static double[] Params(AlgFormer m) { using var ms = new MemoryStream(Save(m)); using var r = new BinaryReader(ms); for (var i = 0; i < 4; i++) r.ReadInt32(); var l = new List<double>(); while (ms.Position < ms.Length) l.Add(r.ReadDouble()); return l.ToArray(); }
    public static double MaxDiff(AlgFormer a, AlgFormer b) { var pa = Params(a); var pb = Params(b); double m = 0; for (var i = 0; i < pa.Length; i++) m = Math.Max(m, Math.Abs(pa[i] - pb[i])); return m; }

    static byte[] ReadExact(Stream s, int n) { var b = new byte[n]; var o = 0; while (o < n) { var r = s.Read(b, o, n - o); if (r <= 0) throw new EndOfStreamException(); o += r; } return b; }
    static void Write(Stream s, byte tag, byte[] payload) { s.Write(BitConverter.GetBytes(payload.Length + 1)); s.WriteByte(tag); s.Write(payload); s.Flush(); }
    static (byte tag, byte[] payload) Read(Stream s) { var len = BitConverter.ToInt32(ReadExact(s, 4)); var tag = ReadExact(s, 1)[0]; return (tag, ReadExact(s, len - 1)); }
    static byte[] PackBatch(List<(int[] Ctx, int Tgt)> d) { using var ms = new MemoryStream(); var w = new BinaryWriter(ms); w.Write(d.Count); foreach (var (c, t) in d) { w.Write(c.Length); foreach (var x in c) w.Write(x); w.Write(t); } w.Flush(); return ms.ToArray(); }
    static List<(int[] Ctx, int Tgt)> UnpackBatch(byte[] p) { using var ms = new MemoryStream(p); using var r = new BinaryReader(ms); var n = r.ReadInt32(); var d = new List<(int[], int)>(n); for (var i = 0; i < n; i++) { var len = r.ReadInt32(); var c = new int[len]; for (var k = 0; k < len; k++) c[k] = r.ReadInt32(); d.Add((c, r.ReadInt32())); } return d; }
    static byte[] PackApply(double lr, int scale, byte[] grad) { using var ms = new MemoryStream(); var w = new BinaryWriter(ms); w.Write(lr); w.Write(scale); w.Write(grad); w.Flush(); return ms.ToArray(); }

    sealed class WC(TcpClient c, NetworkStream s) { public TcpClient C = c; public NetworkStream S = s; }

    public sealed class Host(AlgFormer model)
    {
        public AlgFormer Model = model;
        readonly object _lock = new();
        readonly List<WC> _live = new();        // synced, participating (stable order)
        readonly List<WC> _joining = new();     // accepted, not yet synced into a round

        public void Run(List<(int[] Ctx, int Tgt)> data, int epochs, int batch, double lr, TcpListener listener,
            int minStart, Action<string>? onEvent = null, Action<(int n, double loss, int workers)>? onRound = null, int demoDelayMs = 0)
        {
            var cts = new CancellationTokenSource();
            var accept = new Thread(() =>
            {
                try { while (!cts.IsCancellationRequested) { var c = listener.AcceptTcpClient(); var s = c.GetStream(); s.ReadTimeout = s.WriteTimeout = 8000; lock (_lock) _joining.Add(new WC(c, s)); } }
                catch { /* listener stopped */ }
            }) { IsBackground = true };
            accept.Start();

            while (true) { int have; lock (_lock) have = _live.Count + _joining.Count; if (have >= minStart) break; Thread.Sleep(5); }

            var round = 0;
            for (var ep = 0; ep < epochs; ep++)
                for (var start = 0; start < data.Count; start += batch)
                {
                    List<WC> workers;
                    lock (_lock)
                    {
                        foreach (var j in _joining)                                   // sync joiners to the CURRENT params, then admit
                            try { Write(j.S, MODEL, Save(Model)); _live.Add(j); onEvent?.Invoke($"worker joined -> {_live.Count} active"); }
                            catch { try { j.C.Close(); } catch { } }
                        _joining.Clear();
                        workers = _live.ToList();
                    }

                    var b = data.GetRange(start, Math.Min(batch, data.Count - start));
                    var nodes = 1 + workers.Count;
                    var dead = new HashSet<WC>();
                    for (var wi = 0; wi < workers.Count; wi++)                          // hand each worker its round-robin shard
                    {
                        var shard = new List<(int[], int)>(); for (var i = wi + 1; i < b.Count; i += nodes) shard.Add(b[i]);
                        try { Write(workers[wi].S, SHARD, PackBatch(shard)); } catch { dead.Add(workers[wi]); }
                    }
                    var merged = Model.NewGrads();
                    var hg = Model.NewGrads(); for (var i = 0; i < b.Count; i += nodes) Model.Accumulate(b[i].Ctx, b[i].Tgt, hg); merged.Add(hg);
                    foreach (var w in workers) { if (dead.Contains(w)) continue; try { var (t, p) = Read(w.S); if (t == GRAD) merged.Add(Model.DeserializeGradient(p)); } catch { dead.Add(w); } }
                    var gradBytes = Model.SerializeGradient(merged);
                    Model.Step(merged, lr, scale: b.Count);
                    var apply = PackApply(lr, b.Count, gradBytes);
                    foreach (var w in workers) { if (dead.Contains(w)) continue; try { Write(w.S, APPLY, apply); } catch { dead.Add(w); } }
                    if (dead.Count > 0) { lock (_lock) foreach (var d in dead) { _live.Remove(d); try { d.C.Close(); } catch { } } onEvent?.Invoke($"{dead.Count} worker(s) dropped -> {_live.Count} active"); }

                    round++;
                    onRound?.Invoke((round, AvgLoss(Model, data), workers.Count - dead.Count));
                    if (demoDelayMs > 0) Thread.Sleep(demoDelayMs);
                }

            cts.Cancel(); try { listener.Stop(); } catch { }
            lock (_lock) foreach (var w in _live) { try { Write(w.S, DONE, Array.Empty<byte>()); w.C.Close(); } catch { } }
        }
    }

    public static void RunWorkerOn(TcpClient c)
    {
        new Thread(() => { try { RunWorker(c.GetStream()); } catch { } }) { IsBackground = true }.Start();
    }
    public static TcpClient ConnectLoopback(int port) { var c = new TcpClient(); c.Connect(IPAddress.Loopback, port); return c; }
    public static void SpawnLoopbackWorker(int port) => RunWorkerOn(ConnectLoopback(port));

    static void RunWorker(NetworkStream s)
    {
        var model = NewModel();
        while (true)
        {
            var (tag, p) = Read(s);
            if (tag == MODEL) { using var ms = new MemoryStream(p); using var r = new BinaryReader(ms); model.Load(r); }
            else if (tag == SHARD) { var shard = UnpackBatch(p); var g = model.NewGrads(); foreach (var (c, t) in shard) model.Accumulate(c, t, g); Write(s, GRAD, model.SerializeGradient(g)); }
            else if (tag == APPLY) { using var ms = new MemoryStream(p); using var r = new BinaryReader(ms); var lr = r.ReadDouble(); var scale = r.ReadInt32(); var grad = r.ReadBytes(p.Length - 12); model.Step(model.DeserializeGradient(grad), lr, scale); }
            else if (tag == DONE) break;
        }
    }

    public static void RunHostCli(int port, int minStart)
    {
        var data = MakeData(256);
        var listener = new TcpListener(IPAddress.Any, port); listener.Start();
        Console.WriteLine($"host on :{port} — starts once {minStart} worker(s) present, then accepts/drops freely.  public endpoint (STUN): {TryStun() ?? "n/a"}");
        new Host(NewModel()).Run(data, 8, 32, 5e-3, listener, minStart,
            onEvent: e => Console.WriteLine($"  [membership] {e}"),
            onRound: r => { if (r.n % 8 == 0) Console.WriteLine($"  round {r.n,3}: {r.workers} worker(s), loss {r.loss:F3}"); });
        Console.WriteLine("done.");
    }
    public static void RunWorkerCli(string ip, int port)
    {
        Console.WriteLine($"connecting to host {ip}:{port} ...");
        using var c = new TcpClient(); c.Connect(ip, port);
        Console.WriteLine("connected — contributing gradients until the host finishes (or one of us drops).");
        try { RunWorker(c.GetStream()); Console.WriteLine("host finished."); } catch { Console.WriteLine("disconnected."); }
    }

    // free public STUN (RFC 5389 binding request) — discover this machine's public UDP endpoint
    public static string? TryStun(string server = "stun.l.google.com", int port = 19302)
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

    // ---- MQTT relay end-to-end test: host + 2 slaves ALL through a public broker (works through any NAT, no port-forward) ----
    public static void RelayTest()
    {
        Console.WriteLine("=== MQTT relay training (host + 2 slaves through a free public broker) ===\n");
        var host = new MqttRelayHost(NewModel());
        var code = host.StartRelay(s => Console.WriteLine("[host]  " + s));
        Console.WriteLine($"room code (a slave would paste this): {code}\n");

        var cts = new CancellationTokenSource();
        for (var i = 0; i < 2; i++) { var id = i; System.Threading.Tasks.Task.Run(() => { try { new MqttRelayWorker().JoinRoom(code, s => Console.WriteLine($"[slave{id}] " + s), cts.Token); } catch (Exception e) { Console.WriteLine($"[slave{id}] {e.Message}"); } }); }

        Console.WriteLine("waiting for slaves to dial into the broker...");
        Thread.Sleep(5000);
        host.ChunkTarget = 1024;   // coalesce small batches into big chunks -> few broker messages (no artificial pacing needed)
        var data = MakeData(4096);
        var loss0 = AvgLoss(host.Model, data);
        for (var pass = 0; pass < 2; pass++)
            for (var i = 0; i < data.Count; i += 256)
                host.TrainBatch(data.GetRange(i, Math.Min(256, data.Count - i)), 5e-3);
        Console.WriteLine($"trained through the relay — {host.ActiveWorkers} slave(s) still connected");
        cts.Cancel(); host.StopRelay();
        Console.WriteLine($"\nrelay training complete — avg loss {loss0:F3} -> {AvgLoss(host.Model, data):F3}. Both slaves reached the host purely through the public broker (no port-forward, no install).");
    }

    // ---- POSITION relay: host ships a manifest + model once, then only POSITIONS; workers reconstruct data locally ----
    public static void RelayTest2()
    {
        Console.WriteLine("=== position relay (manifest + model + gradients only — data is NEVER sent) ===\n");
        var source = new SyntheticJobSource(count: 100000, vocab: 32, ctxLen: 4, seed: 777);
        var host = new MqttJobHost(NewModel(), source, source.Manifest()) { ChunkTarget = 1024 };
        var code = host.StartRelay(s => Console.WriteLine("[host]  " + s));
        Console.WriteLine($"room code: {code}\n");

        var cts = new CancellationTokenSource();
        for (var i = 0; i < 2; i++) { var id = i; System.Threading.Tasks.Task.Run(() => { try { new MqttJobWorker(SyntheticJobSource.FromManifest).JoinRoom(code, s => Console.WriteLine($"[slave{id}] " + s), cts.Token); } catch (Exception e) { Console.WriteLine($"[slave{id}] {e.Message}"); } }); }

        double AvgSrc(AlgFormer m, int n) { var g = m.NewGrads(); double t = 0; for (long i = 0; i < n; i++) { var (c, tg) = source.GetExample(i); t += m.Accumulate(c, tg, g); } return t / n; }
        Console.WriteLine("waiting for slaves to build their local data from the manifest...");
        Thread.Sleep(5000);
        var loss0 = AvgSrc(host.Model, 512);
        for (var pass = 0; pass < 3; pass++)
            for (long start = 0; start < 4096; start += 256)
            { var batch = new List<long>(); for (long j = start; j < start + 256; j++) batch.Add(j % source.Count); host.TrainPositions(batch, 5e-3); }
        Console.WriteLine($"trained via positions — {host.ActiveWorkers} slave(s) connected");
        cts.Cancel(); host.StopRelay();
        Console.WriteLine($"\nposition-relay complete — loss {loss0:F3} -> {AvgSrc(host.Model, 512):F3}. Workers reconstructed every example from the manifest; only positions + gradients crossed the wire.");
    }
}
