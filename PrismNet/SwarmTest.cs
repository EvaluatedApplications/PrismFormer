using System.Net;
using System.Net.Sockets;
using PrismFormer;

// ═══════════════════════════════════════════════════════════════════════════════════════════════════════════
//  SwarmTest — a small-scale integration test for the PrismFormer swarm (see SWARM.md).
//
//  Spins up 1 host + N tiny PrismFormer "cells" over REAL loopback TCP sockets. Every cell holds the WHOLE model.
//  The host ships the manifest (once) + model (once), then each round ships only example POSITIONS (indices) and
//  merges the gradients that come back. Each cell reconstructs its assigned examples LOCALLY from the manifest —
//  the training DATA never crosses the wire, only positions + gradients + the initial model.
//
//  It asserts the two claims that make the colony real:
//    (1) CONVERGENCE  — the swarm-trained model is BIT-FOR-BIT identical to a single-node reference.
//    (2) COHERENCE    — every cell (host + all workers) ends on byte-identical weights.
//
//  Deterministic and dependency-free (no external broker): a real integration test, runnable as `prismnet swarmtest`.
// ═══════════════════════════════════════════════════════════════════════════════════════════════════════════

static class SwarmTest
{
    // A deliberately TINY prismformer — this test exercises the network/merge mechanics, not model capacity.
    const int VOCAB = 32, SHIFTS = 8, LAYERS = 2, MAXCTX = 4, DMODEL = 32, SEED = 1;
    const int PASSES = 2, BATCH = 64, WINDOW = 2048;   // train positions 0..WINDOW over PASSES passes, in minibatches
    const double LR = 5e-3;

    const byte MANIFEST = 1, MODEL = 2, SHARD = 3, GRAD = 4, APPLY = 5, DONE = 6;

    static AlgFormer NewModel() => new(VOCAB, shifts: SHIFTS, layers: LAYERS, maxContext: MAXCTX, dModel: DMODEL, frozenPrefix: 0, embedSeed: null, seed: SEED);

    public static void Run(int workers)
    {
        if (workers < 1) workers = 1;
        Console.WriteLine("=== PrismFormer swarm integration test (positions only — data never sent) ===\n");
        Console.WriteLine($"cells: 1 host + {workers} worker(s), all holding the whole model  |  tiny model: vocab {VOCAB}, d {DMODEL}, S {SHIFTS}, L {LAYERS}\n");

        var source = new SyntheticJobSource(count: 8192, vocab: VOCAB, ctxLen: MAXCTX, seed: 777);
        var manifest = source.Manifest();
        var plan = BuildPlan();   // the identical stream of position-minibatches every configuration will train on
        var nodes = 1 + workers;

        // ---------- single-node REFERENCE: what the swarm must reproduce exactly ----------
        var reference = NewModel();
        var loss0 = AvgLoss(reference, source, WINDOW);
        foreach (var batch in plan)
        {
            var merged = reference.NewGrads();
            for (var k = 0; k < nodes; k++)   // node k owns the round-robin stride k, k+nodes, ...  (host = k0)
            {
                var g = reference.NewGrads();
                for (var i = k; i < batch.Count; i += nodes) { var (c, t) = source.GetExample(batch[i]); reference.Accumulate(c, t, g); }
                merged.Add(g);
            }
            reference.Step(merged, LR, scale: batch.Count);
        }
        var refLoss = AvgLoss(reference, source, WINDOW);

        // ---------- the SWARM over loopback TCP ----------
        var listener = new TcpListener(IPAddress.Loopback, 0); listener.Start();
        var port = ((IPEndPoint)listener.LocalEndpoint).Port;

        var cells = new List<Worker>();
        var threads = new List<Thread>();
        for (var i = 0; i < workers; i++)
        {
            var c = new TcpClient(); c.Connect(IPAddress.Loopback, port);
            var w = new Worker(SyntheticJobSource.FromManifest, c.GetStream());
            var t = new Thread(w.Loop) { IsBackground = true }; t.Start();
            cells.Add(w); threads.Add(t);
        }

        var host = new Host(NewModel(), source);
        host.Run(plan, listener, minStart: workers);
        foreach (var t in threads) t.Join(5000);
        listener.Stop();

        // ---------- verdict ----------
        var refBytes = Save(reference);
        var hostOk = Save(host.Model).SequenceEqual(refBytes);
        var workerOks = cells.Select(c => c.Model is { } m && Save(m).SequenceEqual(refBytes)).ToList();
        var coherent = hostOk && workerOks.All(x => x);
        var swarmLoss = AvgLoss(host.Model, source, WINDOW);

        Console.WriteLine($"rounds trained            : {plan.Count}");
        Console.WriteLine($"wire traffic              : model {host.ModelBytes:N0} B (once) + positions {host.PosBytes:N0} B + gradients {host.GradBytes:N0} B   |   training DATA sent: 0 B");
        Console.WriteLine($"real training happened    : avg loss {loss0:F3} -> {swarmLoss:F3}  (reference {refLoss:F3})\n");
        Console.WriteLine($"(1) CONVERGENCE — swarm bit-for-bit identical to single-node : {(hostOk ? "PASS" : "FAIL")}   (max param diff {MaxDiff(host.Model, reference):E1})");
        Console.WriteLine($"(2) COHERENCE   — all {workers} worker cell(s) byte-identical  : {(workerOks.All(x => x) ? "PASS" : "FAIL")}   ({workerOks.Count(x => x)}/{workers} agree)");
        Console.WriteLine();

        var pass = coherent && swarmLoss < loss0;
        Console.WriteLine(pass
            ? $"RESULT: PASS — {nodes} cells trained as one colony over the network; positions only, data never left home."
            : "RESULT: FAIL");
        if (!pass) Environment.ExitCode = 1;
    }

    // The fixed schedule of position-minibatches, identical for reference and swarm.
    static List<List<long>> BuildPlan()
    {
        var plan = new List<List<long>>();
        for (var pass = 0; pass < PASSES; pass++)
            for (var start = 0; start < WINDOW; start += BATCH)
            {
                var b = new List<long>(); for (var j = start; j < start + BATCH && j < WINDOW; j++) b.Add(j);
                plan.Add(b);
            }
        return plan;
    }

    static double AvgLoss(AlgFormer m, IJobSource src, int n)
    {
        var g = m.NewGrads(); double s = 0; for (long i = 0; i < n; i++) { var (c, t) = src.GetExample(i); s += m.Accumulate(c, t, g); } return s / n;
    }
    static byte[] Save(AlgFormer m) { using var ms = new MemoryStream(); var w = new BinaryWriter(ms); m.Save(w); w.Flush(); return ms.ToArray(); }
    static double[] Params(byte[] saved) { using var ms = new MemoryStream(saved); using var r = new BinaryReader(ms); for (var i = 0; i < 4; i++) r.ReadInt32(); var l = new List<double>(); while (ms.Position < ms.Length) l.Add(r.ReadDouble()); return l.ToArray(); }
    static double MaxDiff(AlgFormer a, AlgFormer b) { var pa = Params(Save(a)); var pb = Params(Save(b)); double m = 0; for (var i = 0; i < Math.Min(pa.Length, pb.Length); i++) m = Math.Max(m, Math.Abs(pa[i] - pb[i])); return m; }

    // ---- length-prefixed framing over a stream ----
    static byte[] ReadExact(Stream s, int n) { var b = new byte[n]; var o = 0; while (o < n) { var r = s.Read(b, o, n - o); if (r <= 0) throw new EndOfStreamException(); o += r; } return b; }
    static void Write(Stream s, byte tag, byte[] payload) { s.Write(BitConverter.GetBytes(payload.Length + 1)); s.WriteByte(tag); s.Write(payload); s.Flush(); }
    static (byte tag, byte[] payload) Read(Stream s) { var len = BitConverter.ToInt32(ReadExact(s, 4)); var tag = ReadExact(s, 1)[0]; return (tag, ReadExact(s, len - 1)); }
    static byte[] PackPositions(List<long> p) { using var ms = new MemoryStream(); var w = new BinaryWriter(ms); w.Write(p.Count); foreach (var x in p) w.Write(x); w.Flush(); return ms.ToArray(); }
    static List<long> UnpackPositions(byte[] p) { using var r = new BinaryReader(new MemoryStream(p)); var n = r.ReadInt32(); var l = new List<long>(n); for (var i = 0; i < n; i++) l.Add(r.ReadInt64()); return l; }
    static byte[] PackApply(double lr, int scale, byte[] grad) { using var ms = new MemoryStream(); var w = new BinaryWriter(ms); w.Write(lr); w.Write(scale); w.Write(grad); w.Flush(); return ms.ToArray(); }

    sealed class Conn(TcpClient c, NetworkStream s) { public readonly TcpClient C = c; public readonly NetworkStream S = s; }

    // ---- HOST cell: owns the manifest + model, ships positions, merges gradients exactly ----
    sealed class Host(AlgFormer model, IJobSource source)
    {
        public AlgFormer Model = model;
        public long ModelBytes, PosBytes, GradBytes;
        readonly object _lock = new();
        readonly List<Conn> _live = new(), _joining = new();

        public void Run(List<List<long>> plan, TcpListener listener, int minStart)
        {
            var cts = new CancellationTokenSource();
            var accept = new Thread(() => { try { while (!cts.IsCancellationRequested) { var c = listener.AcceptTcpClient(); var s = c.GetStream(); s.ReadTimeout = s.WriteTimeout = 10000; lock (_lock) _joining.Add(new Conn(c, s)); } } catch { } }) { IsBackground = true };
            accept.Start();
            while (true) { int have; lock (_lock) have = _live.Count + _joining.Count; if (have >= minStart) break; Thread.Sleep(5); }

            var manifest = ((SyntheticJobSource)source).Manifest();
            foreach (var batch in plan)
            {
                List<Conn> workers;
                lock (_lock)
                {
                    foreach (var j in _joining)   // a joiner gets the manifest + current model, then joins the round
                    {
                        try { Write(j.S, MANIFEST, manifest); var mb = Model.Serialize(); Write(j.S, MODEL, mb); ModelBytes += mb.Length; _live.Add(j); }
                        catch { try { j.C.Close(); } catch { } }
                    }
                    _joining.Clear();
                    workers = _live.ToList();
                }

                var nodes = 1 + workers.Count;
                var dead = new HashSet<Conn>();
                for (var wi = 0; wi < workers.Count; wi++)   // worker wi owns stride (wi+1)
                {
                    var shard = new List<long>(); for (var i = wi + 1; i < batch.Count; i += nodes) shard.Add(batch[i]);
                    var pb = PackPositions(shard); PosBytes += pb.Length;
                    try { Write(workers[wi].S, SHARD, pb); } catch { dead.Add(workers[wi]); }
                }

                var merged = Model.NewGrads();
                var hg = Model.NewGrads();   // host owns stride 0
                for (var i = 0; i < batch.Count; i += nodes) { var (c, t) = source.GetExample(batch[i]); Model.Accumulate(c, t, hg); }
                merged.Add(hg);
                foreach (var w in workers) { if (dead.Contains(w)) continue; try { var (t, p) = Read(w.S); if (t == GRAD) { GradBytes += p.Length; merged.Add(Model.DeserializeGradient(p)); } } catch { dead.Add(w); } }

                Model.Step(merged, LR, scale: batch.Count);
                var apply = PackApply(LR, batch.Count, Model.SerializeGradient(merged));
                foreach (var w in workers) { if (dead.Contains(w)) continue; try { Write(w.S, APPLY, apply); } catch { dead.Add(w); } }
                if (dead.Count > 0) lock (_lock) foreach (var d in dead) { _live.Remove(d); try { d.C.Close(); } catch { } }
            }

            cts.Cancel(); try { listener.Stop(); } catch { }
            lock (_lock) foreach (var w in _live) { try { Write(w.S, DONE, Array.Empty<byte>()); } catch { } }
        }
    }

    // ---- WORKER cell: builds its source from the manifest, reconstructs examples locally, contributes gradients ----
    sealed class Worker(Func<byte[], IJobSource> sourceFactory, NetworkStream stream)
    {
        public AlgFormer? Model;
        IJobSource? _source;

        public void Loop()
        {
            try
            {
                while (true)
                {
                    var (tag, p) = Read(stream);
                    if (tag == MANIFEST) _source = sourceFactory(p);
                    else if (tag == MODEL) Model = AlgFormer.Deserialize(p);
                    else if (tag == SHARD)
                    {
                        var positions = UnpackPositions(p);
                        var g = Model!.NewGrads();
                        foreach (var pos in positions) { var (c, t) = _source!.GetExample(pos); Model.Accumulate(c, t, g); }
                        Write(stream, GRAD, Model.SerializeGradient(g));
                    }
                    else if (tag == APPLY)
                    {
                        using var r = new BinaryReader(new MemoryStream(p));
                        var lr = r.ReadDouble(); var scale = r.ReadInt32(); var grad = r.ReadBytes(p.Length - 12);
                        Model!.Step(Model.DeserializeGradient(grad), lr, scale);
                    }
                    else if (tag == DONE) break;
                }
            }
            catch { /* dropped — the host continues without us */ }
        }
    }
}
