// Copyright (c) 2026 Dongyang Stephen Chen, trading as Evaluated Applications. All rights reserved.
// Source-available for study and verification; commercial use, redistribution, or derivatives require permission. See LICENSE.

using PrismFormer;

namespace PrismGym;

/// <summary>
/// A HEADLESS Prism node (see SWARM.md) — Prism Studio's Model+Network behaviour with no window. It joins an existing
/// swarm by room code and participates on BOTH channels, exactly like MainForm does when you paste a code:
///   • relay WORKER (<see cref="MqttRelayWorker"/>) — donates gradients so the HOST's head trains across every peer
///     ("it adds up"): the host slices each batch out, we return gradients, the host sums them bit-exact.
///   • <see cref="SwarmChatter"/> — answers the swarm's REPL escalations off the same frozen-spec head, and bleeds
///     (shares pairs + a tiny weight-slice elastic average + bleed-chat: ask a peer, learn its reply).
/// Frozen PRISM-1 spec, so it merges bit-exact with any other node. Ctrl-C stops it (saves its checkpoint).
/// Usage: <c>prismgym headless &lt;roomCode&gt; [dataDir]</c>
/// </summary>
public static class HeadlessNode
{
    public static int Run(string code, string? dataDir)
    {
        if (string.IsNullOrWhiteSpace(code)) code = MqttRelay.MakeRoomCode(MqttRelay.DefaultRoom);   // no code → auto-join the colony
        var room = MqttRelay.ParseRoomCode(code);
        if (room is null) { Console.Error.WriteLine("not a relay code"); return 2; }

        var prismDir = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Prism");
        Directory.CreateDirectory(prismDir);
        dataDir ??= Path.Combine(prismDir, "headless-data");     // no folder given → a private one (worker + bleed only, no local corpus to grind)
        var ckpt = Path.Combine(prismDir, "prism-headless.bin"); // our own checkpoint — never clobbers the Studio's prism.bin

        void Log(string s) => Console.WriteLine($"{DateTime.Now:HH:mm:ss} {s}");

        Log($"headless node — spec {PrismSpec.Signature}");
        var model = new StudioModel(dataDir);
        Log($"model ready — {model.ParamCount:N0} params · data: {dataDir}");
        // seed from an existing checkpoint if the signature matches: our own file first, else the Studio's shared prism.bin
        if (model.Load(ckpt)) Log($"loaded {ckpt}");
        else if (model.Load(Path.Combine(prismDir, "prism.bin"))) Log("seeded from the Studio checkpoint (prism.bin)");
        else Log("fresh model (no matching checkpoint) — will learn from bleed + local data");

        var cts = new CancellationTokenSource();
        Console.CancelKeyPress += (_, e) => { e.Cancel = true; Log("Ctrl-C — stopping…"); cts.Cancel(); };

        // ── BLEED + REPL channel: serve peers' queries off our head; elastic-average incoming weight slices toward the swarm ──
        var chatter = new SwarmChatter(room, model.Serve, model.AbsorbPair,   // AbsorbPair = gossip-persist + backprop-on-wrong (this node trains)
            (start, vals) => Task.Run(() => model.MergeWeightSlice(start, vals, 0.05)), Log,
            onGroup: (h, t) => model.AppendGroup(h, t), groupServe: model.GroupReply, absorbContext: model.AbsorbChat, signature: PrismSpec.Signature);   // group chat: replies round-robin, trains on it, + learns from queries it's asked
        chatter.Start();
        Log("joined swarm room — peers appear as they say hello");

        // ── GRADIENT channel: donate gradients to the host's head (this is how the shared head trains across peers) ──
        var worker = new MqttRelayWorker();
        var workerTask = Task.Run(() =>
        {
            try { worker.JoinRoom(code, Log, cts.Token); }
            catch (Exception e) { Log("[worker] " + e.Message.Split('\n')[0]); }
        });

        // ── BLEED timer — mirrors the Studio: share a pair, bleed a tiny weight slice, and bleed-chat (ask a peer, learn its reply) ──
        var bleed = new System.Threading.Timer(_ =>
        {
            try
            {
                // Bleed is bounded + proximity-routed INSIDE the chatter: each push goes to the K CLOSEST peers (ping-ranked)
                if (model.RandomPair(Random.Shared) is { } p) chatter.SharePair(p.Prompt, p.Target);
                if (model.WeightSlice(Random.Shared, 1024) is { } ws) chatter.ShareWeightSlice(ws.Start, ws.Vals);
                var q = model.RandomPrompt(Random.Shared);   // one of OUR chat contexts
                if (q != null && chatter.AskSwarm(q, 8000) is { } a && a.Continuation.Trim().Length > 0)
                { model.LearnFromPeer(q, a.Continuation.Trim()); Log("[bleed] ↔ distilled a peer's personality on our chat context"); }
            }
            catch (Exception e) { Log("[bleed] " + e.Message.Split('\n')[0]); }
        }, null, 30000, 30000);

        // ── Train our OWN head on whatever local data we have, so our bleed is a real learner (not init noise). Safe if empty. ──
        var trainTask = Task.Run(() =>
        {
            try { model.Train(int.MaxValue, 5.0, _ => { }, Log, cts.Token); }   // runs until Ctrl-C; report discarded, generation samples → Log
            catch (Exception e) { Log("[train] " + e.Message.Split('\n')[0]); }
        });

        var saver = new System.Threading.Timer(_ => { try { model.Save(ckpt); } catch { } }, null, 60000, 60000);   // periodic checkpoint

        cts.Token.WaitHandle.WaitOne();     // block until Ctrl-C
        Log("shutting down…");
        try { bleed.Dispose(); saver.Dispose(); } catch { }
        try { chatter.Stop(); } catch { }
        try { Task.WaitAll(new[] { workerTask, trainTask }, 5000); } catch { }
        try { model.Save(ckpt); Log($"saved {ckpt}"); } catch { }
        Log("bye");
        return 0;
    }

    static string Trunc(string s) => s.Length <= 32 ? s : s[..32] + "…";

    /// <summary>Headless HOST: open a relay room, print the code, and train — fanning each batch out to any workers that
    /// join (gradient-sum across peers). Complements <see cref="Run"/> (the worker). Usage: <c>prismgym host [dataDir]</c></summary>
    public static int Host(string? dataDir)
    {
        var prismDir = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Prism");
        Directory.CreateDirectory(prismDir);
        dataDir ??= Path.Combine(prismDir, "headless-data");
        var ckpt = Path.Combine(prismDir, "prism-host.bin");   // own file — never clobbers the Studio's prism.bin
        void Log(string s) => Console.WriteLine($"{DateTime.Now:HH:mm:ss} {s}");

        var model = new StudioModel(dataDir);
        Log($"headless HOST — spec {PrismSpec.Signature} · {model.ParamCount:N0} params · data: {dataDir}");
        if (model.Load(ckpt)) Log($"loaded {ckpt}");
        else if (model.Load(Path.Combine(prismDir, "prism.bin"))) Log("seeded from the Studio checkpoint (prism.bin)");

        var code = model.EnableHosting(m => Log("[relay] " + m));
        Log($"RELAY CODE: {code}");
        Log($"join from another machine with:  prismgym headless {code}");

        var cts = new CancellationTokenSource();
        Console.CancelKeyPress += (_, e) => { e.Cancel = true; Log("Ctrl-C — stopping…"); cts.Cancel(); };
        var saver = new System.Threading.Timer(_ => { try { model.Save(ckpt); } catch { } }, null, 60000, 60000);
        try { model.Train(int.MaxValue, 5.0, _ => { }, Log, cts.Token); }
        catch (Exception e) { Log("[train] " + e.Message.Split('\n')[0]); }
        try { saver.Dispose(); model.Save(ckpt); } catch { }
        Log("stopped");
        return 0;
    }

    /// <summary>ANCHOR: an always-on, PASSIVE swarm member (for a cheap always-on box next to the broker). It joins the
    /// mesh, bleeds (shares + absorbs weight-slices/pairs), and answers REPL queries — but it does NOT grind a curriculum
    /// and is NOT the gradient-training host. It "passively trains" by absorbing the weight-slices peers bleed to it, so
    /// it drifts toward the swarm without ever touching the corpus. Seed it from a good prism.bin for solid answers.
    /// A STABLE room (a name, not a random GUID) means the swarm always reconvenes at the same address across restarts.
    /// Usage: <c>prismgym anchor &lt;roomCode|roomName&gt;</c>  (or set PRISM_ROOM). Point PRISM_BROKER at the local broker.</summary>
    public static int Anchor(string roomArg)
    {
        var room = MqttRelay.ParseRoomCode(roomArg) ?? (string.IsNullOrWhiteSpace(roomArg) ? MqttRelay.DefaultRoom : roomArg);   // no arg → the colony room
        var code = MqttRelay.MakeRoomCode(room);

        var prismDir = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Prism");
        Directory.CreateDirectory(prismDir);
        var ckpt = Path.Combine(prismDir, "prism-anchor.bin");
        void Log(string s) => Console.WriteLine($"{DateTime.Now:HH:mm:ss} {s}");

        var model = new StudioModel(Path.Combine(prismDir, "anchor-data"));
        Log($"anchor node — spec {PrismSpec.Signature} · {model.ParamCount:N0} params · broker {MqttRelay.Broker}:{MqttRelay.Port}");
        if (model.Load(ckpt)) Log($"loaded {ckpt}");
        else if (model.Load(Path.Combine(prismDir, "prism.bin"))) Log("seeded from prism.bin");
        else Log("fresh model — drop a trained prism.bin next to me for good answers");
        Log($"SWARM CODE (paste into every node): {code}");

        var cts = new CancellationTokenSource();
        Console.CancelKeyPress += (_, e) => { e.Cancel = true; Log("Ctrl-C — stopping…"); cts.Cancel(); };

        var chatter = new SwarmChatter(room,
            // DECISION: the anchor is a PURELY PASSIVE relay — it does NOT answer queries and does NOT train. On this box
            // (1 vCPU) the backprop-on-wrong path ran a full Predict forward on every bled pair AND every query-context pair,
            // unbounded by the mesh rate — that pinned the core at 100% and starved sshd (couldn't even deploy). So the anchor
            // now only: relays gossip, averages in peers' weights (cheap, no gradients), and carries group-chat context.
            // The trained PEERS do the learning; the anchor just keeps the mesh discoverable and the consensus weights flowing.
            _ => (0.0, ""),
            model.AddGossipPair,   // relay-only: persist + gossip the bled pair onward — NO forward pass, NO backprop
            (start, vals) => Task.Run(() => model.MergeWeightSlice(start, vals, 0.05)), Log,
            onGroup: (h, t) => model.AppendGroup(h, t), signature: PrismSpec.Signature);   // carries group-chat context; does NOT learn from it
        chatter.Start();
        Log("joined the mesh — PASSIVE relay: gossip + weight-average only (no answering, no training — CPU stays free)");

        var bleed = new System.Threading.Timer(_ =>
        {
            try   // PASSIVE but ACTIVE-DRIVER: share what we ALREADY have + drive the chat by querying peers. No gradient steps — the anchor never trains.
            {
                Log($"[mesh] {chatter.PeerCount} peer(s) nearby");
                if (model.WeightSlice(Random.Shared, 1024) is { } ws) chatter.ShareWeightSlice(ws.Start, ws.Vals);
                if (model.RandomPair(Random.Shared) is { } p) chatter.SharePair(p.Prompt, p.Target);
                // Always-on chat driver: take a natural prefix of the GROUP chat context we've been accumulating and ask a
                // peer "what would you say here?" — like the worker's bleed-chat, EXCEPT we don't distil the reply (that's
                // training). Cheap for us (we only SEND the ask); the PEER, being asked, learns from the human turns the
                // context carries (_absorbContext on the query side). So the anchor keeps the group conversation — and
                // everyone else's learning — ticking during idle time without ever touching the corpus or a gradient.
                var q = model.RandomPrompt(Random.Shared);
                if (!string.IsNullOrWhiteSpace(q) && chatter.AskSwarm(q, 8000) is { } a && a.Continuation.Trim().Length > 0)
                    Log("[chat] asked a peer to continue the group context (it learns from the human turns; we don't distil)");
            }
            catch (Exception e) { Log("[bleed] " + e.Message.Split('\n')[0]); }
        }, null, 30000, 30000);
        var saver = new System.Threading.Timer(_ => { try { model.Save(ckpt); } catch { } }, null, 120000, 120000);

        cts.Token.WaitHandle.WaitOne();
        Log("shutting down…");
        try { bleed.Dispose(); saver.Dispose(); chatter.Stop(); model.Save(ckpt); Log($"saved {ckpt}"); } catch { }
        Log("bye");
        return 0;
    }

    /// <summary>One-shot mesh query: join the room, ask the nearest peers a prompt, print the most-confident answer, exit.
    /// Lets you (or me) chat to the running swarm from the command line. Usage: <c>prismgym ask &lt;roomCode&gt; &lt;prompt…&gt;</c></summary>
    public static int Ask(string code, string prompt)
    {
        var room = MqttRelay.ParseRoomCode(code);
        if (room is null) { Console.Error.WriteLine("not a relay code"); return 2; }
        if (string.IsNullOrWhiteSpace(prompt)) { Console.Error.WriteLine("nothing to ask — prismgym ask <code> <prompt>"); return 2; }

        // an asker-only node: we don't serve, bleed, or answer others — just find peers and ask
        var chatter = new SwarmChatter(room, _ => (0.0, ""), (_, _) => { }, null, Console.WriteLine, signature: PrismSpec.Signature);
        chatter.Start();
        Console.WriteLine("finding peers…");
        for (var i = 0; i < 16 && chatter.PeerCount == 0; i++) Thread.Sleep(500);   // wait for HELLO → roster → discovery
        if (chatter.PeerCount == 0) { Console.WriteLine("no peers in the room"); chatter.Stop(); return 1; }
        Console.WriteLine($"asking {chatter.PeerCount} peer(s): \"{prompt}\"");
        var a = chatter.AskSwarm(prompt, 8000);
        Console.WriteLine(a is { } ans ? $"\n→ \"{ans.Continuation}\"   (confidence {ans.Conf:F2})" : "\n→ no answer");
        chatter.Stop();
        return 0;
    }
}
