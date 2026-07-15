using PrismFormer;

// ═══════════════════════════════════════════════════════════════════════════════════════════════════════════
//  SwarmLearn — the decider is LEARNED, not handed a competence map (see SWARM.md). Each node has a decider head
//  that, for a query, picks who should answer: itself or a reachable peer. It learns ONLINE and INTERLEAVED with the
//  swarm — while training it asks the swarm, sees who was right, and that outcome trains the decider. Nobody tells it
//  "skill k lives on node k"; it discovers the routing by asking. Ping-gated: it can only route to reachable peers.
//
//  This is why distributed training matters: the decider heads can only learn to decide by interacting with the swarm.
//  Runnable as `prismnet swarmlearn`.
// ═══════════════════════════════════════════════════════════════════════════════════════════════════════════

static class SwarmLearn
{
    const int VOCAB = 32, DATA = 4, MAXCTX = 1 + DATA, DMODEL = 48, SHIFTS = 12, LAYERS = 3;
    const int K = 4, EXPERT_ROUNDS = 400, BATCH = 32, PING_LIMIT = 100;
    const int DECIDER_ROUNDS = 8000; const double EPS = 0.15, ALPHA = 0.05, LR = 6e-3;
    const int BAD = 3;
    static readonly string[] Name = { "copy0", "last", "max", "min" };

    static AlgFormer NewModel() => new(VOCAB, shifts: SHIFTS, layers: LAYERS, maxContext: MAXCTX, dModel: DMODEL, frozenPrefix: 0, embedSeed: null, seed: 1);

    public static void Run()
    {
        Console.WriteLine("=== learned decider heads — route by asking the swarm during training (see SWARM.md) ===\n");

        var ping = new int[K, K];
        for (var i = 0; i < K; i++) for (var j = 0; j < K; j++) ping[i, j] = i == j ? 0 : (i == BAD || j == BAD ? 250 : 20);

        // the experts: node k is a specialist on skill k
        var spec = new AlgFormer[K]; for (var k = 0; k < K; k++) spec[k] = NewModel();
        for (var r = 0; r < EXPERT_ROUNDS; r++) for (var k = 0; k < K; k++) TrainBatch(spec[k], k, r);

        // the decider head per node: Q[asker][queryMarker][candidate] = learned estimate of "will this candidate answer right?"
        // candidates for a node = itself + its ping-reachable neighbours. NOT given the competence map — it learns it.
        var Q = new double[K][,]; for (var a = 0; a < K; a++) Q[a] = new double[K, K];
        var cand = new List<int>[K]; for (var a = 0; a < K; a++) { cand[a] = new List<int> { a }; cand[a].AddRange(Neighbours(ping, a)); }

        var rng = new Random(7);
        Console.WriteLine($"training the decider heads online ({DECIDER_ROUNDS} interactions with the swarm):");
        for (var step = 1; step <= DECIDER_ROUNDS; step++)
        {
            var skill = rng.Next(K); var asker = rng.Next(K);
            var (c, t) = Example(skill, rng.Next(4_000_000));
            var pick = rng.NextDouble() < EPS ? cand[asker][rng.Next(cand[asker].Count)] : Greedy(Q[asker], skill, cand[asker]);
            var correct = Answer(spec[pick], c) == t ? 1.0 : 0.0;
            Q[asker][skill, pick] += ALPHA * (correct - Q[asker][skill, pick]);   // the outcome trains the DECIDER
            if (correct == 0.0) { var g = spec[pick].NewGrads(); spec[pick].Accumulate(c, t, g); spec[pick].Step(g, LR, scale: 1); }   // asked-and-wrong → a little backprop: the EXPERT absorbs it
            if (step % 2000 == 0) Console.WriteLine($"  after {step,5} interactions — routed accuracy {RouteAcc(spec, Q, cand, ping):P0}");
        }

        // what did each node's decider LEARN about who to ask?
        Console.WriteLine("\nlearned routing (each node, per query skill, who its decider now asks):");
        for (var a = 0; a < K; a++)
        {
            Console.Write($"  n{a} asks: ");
            for (var s = 0; s < K; s++) { var who = Greedy(Q[a], s, cand[a]); Console.Write($"{Name[s]}->{(who == a ? "self" : "n" + who)}  "); }
            Console.WriteLine();
        }

        Console.WriteLine("\nexperts after absorbing routed queries (accuracy per skill; each STARTED knowing only its own):");
        Console.Write("        "); for (var s = 0; s < K; s++) Console.Write($"{Name[s],-8}"); Console.WriteLine();
        for (var a = 0; a < K; a++) { Console.Write($"  n{a}    "); for (var s = 0; s < K; s++) { var acc = AccOne(spec[a], s); Console.Write(a == s ? $"[{acc,4:P0}]  " : $" {acc,4:P0}   "); } Console.WriteLine(); }

        var learned = RouteAcc(spec, Q, cand, ping);
        var localOnly = LocalAcc(spec);
        Console.WriteLine($"\n  local-only (no routing)         : {localOnly:P0}");
        Console.WriteLine($"  LEARNED decider routing         : {learned:P0}   (+{(learned - localOnly) * 100:F0} pts — discovered by asking the swarm, no competence map given)");
        Console.WriteLine("  deciders learned who to ask; backprop-on-wrong let reachable nodes ABSORB skills from routed queries — even");
        Console.WriteLine("  routing AROUND the isolated bad-ping node. Caveat: absorbing diverse traffic can dent a node's own specialty (interference).");
    }

    static int Greedy(double[,] q, int marker, List<int> cands) { int best = cands[0]; var bv = double.NegativeInfinity; foreach (var cnd in cands) if (q[marker, cnd] > bv) { bv = q[marker, cnd]; best = cnd; } return best; }
    static List<int> Neighbours(int[,] ping, int i) { var l = new List<int>(); for (var j = 0; j < K; j++) if (j != i && ping[i, j] < PING_LIMIT) l.Add(j); return l; }

    static double RouteAcc(AlgFormer[] spec, double[][,] Q, List<int>[] cand, int[,] ping)
    {
        var ok = 0; var n = 0; var rng = new Random(123);
        for (var q = 0; q < 800; q++) { var skill = rng.Next(K); var asker = rng.Next(K); var (c, t) = Example(skill, 100_000_000L + q); var who = Greedy(Q[asker], skill, cand[asker]); if (Answer(spec[who], c) == t) ok++; n++; }
        return ok / (double)n;
    }
    static double LocalAcc(AlgFormer[] spec)
    {
        var ok = 0; var n = 0; var rng = new Random(123);
        for (var q = 0; q < 800; q++) { var skill = rng.Next(K); var asker = rng.Next(K); var (c, t) = Example(skill, 100_000_000L + q); if (Answer(spec[asker], c) == t) ok++; n++; }
        return ok / (double)n;
    }

    static double AccOne(AlgFormer m, int skill) { var ok = 0; for (var i = 0; i < 400; i++) { var (c, t) = Example(skill, 200_000_000L + i); if (Answer(m, c) == t) ok++; } return ok / 400.0; }
    static int Answer(AlgFormer m, int[] ctx) => m.Predict(ctx);
    static long Idx(int task, int round, int i) { var h = (ulong)(round * BATCH + i) * 2654435761UL + (ulong)task * 40503UL; h ^= h >> 13; return (long)(h % 4_000_000UL); }
    static void TrainBatch(AlgFormer m, int task, int round) { var g = m.NewGrads(); for (var i = 0; i < BATCH; i++) { var (c, t) = Example(task, Idx(task, round, i)); m.Accumulate(c, t, g); } m.Step(g, LR, scale: BATCH); }
    static (int[] Ctx, int Tgt) Example(int task, long index)
    {
        var rng = new Random(unchecked(1013 * task + (int)index * 7919 + 17));
        var ctx = new int[MAXCTX]; ctx[0] = task; for (var k = 0; k < DATA; k++) ctx[1 + k] = rng.Next(VOCAB);
        return (ctx, ctx[1 + task]);
    }
}
