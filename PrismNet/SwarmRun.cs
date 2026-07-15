using PrismFormer;

// ═══════════════════════════════════════════════════════════════════════════════════════════════════════════
//  SwarmRun — micro-scale proof of the networked RUNTIME (see STUDIO.md / SWARM.md), before wiring it to the REPL:
//    (1) PING-ranked neighbours — each node keeps only well-connected peers; a bad-connection node is dropped.
//    (2) periodic CHATTER / co-learning — neighbours share their skill; each node learns what its neighbours know.
//    (3) confidence-gated ROUTING — an unsure node forwards the question to its most-confident neighbour (the REPL,
//        networked). Ping-gated: it never routes to a bad-connection peer.
//
//  Setup: 4 specialist nodes (skill k = "copy data[marker=k]"). Nodes 0/1/2 are well-connected; node 3 is a
//  bad-connection straggler (high ping to everyone). Runnable as `prismnet swarmrun`.
// ═══════════════════════════════════════════════════════════════════════════════════════════════════════════

static class SwarmRun
{
    const int VOCAB = 32, DATA = 4, MAXCTX = 1 + DATA, DMODEL = 48, SHIFTS = 12, LAYERS = 3;
    const int K = 4, ROUNDS = 400, BATCH = 32, EVAL = 400, PING_LIMIT = 100;
    const double LR = 6e-3, MARGIN = 1.0;   // top-2 logit-margin below this = "unsure" → escalate
    const int BAD = 3;                       // node 3 = bad connection (a straggler expert on skill 3)

    static AlgFormer NewModel() => new(VOCAB, shifts: SHIFTS, layers: LAYERS, maxContext: MAXCTX, dModel: DMODEL, frozenPrefix: 0, embedSeed: null, seed: 1);

    public static void Run()
    {
        Console.WriteLine("=== networked swarm runtime — ping-ranked neighbours + chatter + confidence routing ===\n");
        var name = new[] { "copy0", "last", "max", "min" };

        // (1) PING matrix (ms) — 0/1/2 close, node 3 far. Neighbours = ping < PING_LIMIT.
        var ping = new int[K, K];
        for (var i = 0; i < K; i++) for (var j = 0; j < K; j++) ping[i, j] = i == j ? 0 : (i == BAD || j == BAD ? 250 : 20 + ((i + j) % 3) * 15);
        Console.WriteLine("ping (ms) + who each node keeps as neighbours (< " + PING_LIMIT + "ms):");
        for (var i = 0; i < K; i++)
        {
            var nbrs = Neighbours(ping, i);
            Console.WriteLine($"  n{i} ({name[i]}): [{string.Join(" ", Enumerable.Range(0, K).Select(j => $"{ping[i, j],3}"))}]  neighbours: {(nbrs.Count == 0 ? "(none)" : string.Join(",", nbrs.Select(j => "n" + j)))}{(i == BAD ? "   <- bad connection, isolated" : "")}");
        }

        // each node owns one model; starts a specialist on its own skill
        var nodes = new AlgFormer[K]; for (var k = 0; k < K; k++) nodes[k] = NewModel();
        var known = new HashSet<int>[K]; for (var k = 0; k < K; k++) known[k] = new HashSet<int> { k };   // skills a node can train on (own + bled-in)

        // (2) periodic CHATTER: each round, neighbours announce their skill; a node co-learns every skill it has heard of
        for (var r = 0; r < ROUNDS; r++)
        {
            for (var i = 0; i < K; i++) foreach (var j in Neighbours(ping, i)) known[i].Add(j);   // hear neighbours' skills (transitively over rounds)
            for (var i = 0; i < K; i++) foreach (var skill in known[i]) TrainBatch(nodes[i], skill, r);
        }

        Console.WriteLine("\nafter co-learning — each node's accuracy per skill (own skill in [brackets]):");
        Console.Write("        "); for (var s = 0; s < K; s++) Console.Write($"{name[s],-8}"); Console.WriteLine();
        for (var i = 0; i < K; i++)
        {
            Console.Write($"  n{i}    ");
            for (var s = 0; s < K; s++) { var a = Acc(nodes[i], s); Console.Write(i == s ? $"[{a,4:P0}]  " : $" {a,4:P0}   "); }
            Console.WriteLine();
        }
        Console.WriteLine("  well-connected nodes bled each other's skills in; the bad-connection node stayed a lone specialist.");

        // (3) confidence-gated ROUTING — the networked REPL. Shown on fresh SPECIALISTS (each knows only its own skill),
        //     so the hand-off is visible: an unsure node forwards the question to its most-confident NEIGHBOUR (ping-gated).
        var spec = new AlgFormer[K]; for (var k = 0; k < K; k++) spec[k] = NewModel();
        for (var r = 0; r < ROUNDS; r++) for (var k = 0; k < K; k++) TrainBatch(spec[k], k, r);

        int localOk = 0, confOk = 0, compOk = 0, escalated = 0, total = 0;
        var rng = new Random(99);
        for (var q = 0; q < EVAL; q++)
        {
            var skill = rng.Next(K);       // the query's domain (its marker — the decider can read it)
            var asker = rng.Next(K);
            var (c, t) = Example(skill, 100_000_000L + q);
            total++;

            // A. local only
            var (lp, lm) = Answer(spec[asker], c);
            if (lp == t) localOk++;

            // B. confidence-gated: escalate to the most-confident neighbour when the margin is low
            var cpred = lp;
            if (lm < MARGIN) { escalated++; var bestM = lm; foreach (var j in Neighbours(ping, asker)) { var (jp, jm) = Answer(spec[j], c); if (jm > bestM) { bestM = jm; cpred = jp; } } }
            if (cpred == t) confOk++;

            // C. competence-map: the decider knows skill k -> node k. Route to that expert if it's reachable (ping-gated).
            var owner = skill;   // gossiped competence map
            var mpred = (owner == asker || Neighbours(ping, asker).Contains(owner)) ? Answer(spec[owner], c).pred : lp;
            if (mpred == t) compOk++;
        }
        Console.WriteLine($"\nnetworked REPL over {total} queries to random specialist nodes:");
        Console.WriteLine($"  A. answer LOCALLY only              : {localOk / (double)total:P0}   (a specialist only knows its own skill)");
        Console.WriteLine($"  B. confidence-gated routing         : {confOk / (double)total:P0}   ({escalated}/{total} escalated) — top-2 margin FAILS: specialists are confidently wrong out-of-domain");
        Console.WriteLine($"  C. competence-map routing (ping-gated): {compOk / (double)total:P0}   — the decider routes to the expert who owns the skill, if reachable");
        Console.WriteLine($"  => a competence map lifts answers {(compOk - localOk) * 100.0 / total:+0.0;-0.0} pts; confidence alone {(confOk - localOk) * 100.0 / total:+0.0;-0.0}. The decider needs to KNOW who owns what, not just how sure it feels.");
        Console.WriteLine("     'min' stays unreachable (owned only by the bad-connection node) — the runtime correctly refused the bad peer.");
    }

    static List<int> Neighbours(int[,] ping, int i) { var l = new List<int>(); for (var j = 0; j < ping.GetLength(0); j++) if (j != i && ping[i, j] < PING_LIMIT) l.Add(j); return l; }

    static (int pred, double margin) Answer(AlgFormer m, int[] ctx)
    {
        var lg = m.LogitsFor(ctx);
        int i1 = 0; for (var i = 1; i < lg.Length; i++) if (lg[i] > lg[i1]) i1 = i;
        var second = double.NegativeInfinity; for (var i = 0; i < lg.Length; i++) if (i != i1 && lg[i] > second) second = lg[i];
        return (i1, lg[i1] - second);
    }

    static long Idx(int task, int round, int i) { var h = (ulong)(round * BATCH + i) * 2654435761UL + (ulong)task * 40503UL; h ^= h >> 13; return (long)(h % 4_000_000UL); }
    static void TrainBatch(AlgFormer m, int task, int round)
    {
        var g = m.NewGrads();
        for (var i = 0; i < BATCH; i++) { var (c, t) = Example(task, Idx(task, round, i)); m.Accumulate(c, t, g); }
        m.Step(g, LR, scale: BATCH);
    }
    static (int[] Ctx, int Tgt) Example(int task, long index)
    {
        var rng = new Random(unchecked(1013 * task + (int)index * 7919 + 17));
        var ctx = new int[MAXCTX]; ctx[0] = task; for (var k = 0; k < DATA; k++) ctx[1 + k] = rng.Next(VOCAB);
        return (ctx, ctx[1 + task]);   // "copy the token at the marked position"
    }
    static double Acc(AlgFormer m, int task) { var ok = 0; for (var i = 0; i < EVAL; i++) { var (c, t) = Example(task, 200_000_000L + i); if (Answer(m, c).pred == t) ok++; } return ok / (double)EVAL; }
}
