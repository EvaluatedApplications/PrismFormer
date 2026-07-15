using PrismFormer;

// ═══════════════════════════════════════════════════════════════════════════════════════════════════════════
//  SizeReport — capacity planning. Prints real param counts + RAM + per-round wire cost for a few configs, so
//  we can size a node and a coordinator server honestly. Run as `prismnet sizes`.
// ═══════════════════════════════════════════════════════════════════════════════════════════════════════════

static class SizeReport
{
    record Cfg(string Name, int Vocab, int D, int S, int L, int Ctx, int Frozen);

    public static void Run()
    {
        var cfgs = new[]
        {
            new Cfg("tiny (swarmtest)",        32, 32,  8, 2,   4, 0),
            new Cfg("swarmtasks",              32, 48, 12, 3,   5, 0),
            new Cfg("BabyLM base (lean 128)",  95,128, 64, 8, 256, 64),
            new Cfg("bench Mini (fat 512)",    95,512, 32, 4,  16, 64),
        };

        Console.WriteLine("=== PrismFormer size / RAM / wire-cost report ===\n");
        Console.WriteLine($"  {"config",-26}{"params",12}{"weights",11}{"node RAM*",11}{"grad/round",12}");
        Console.WriteLine($"  {"",-26}{"",12}{"(fp64)",11}{"(train)",11}{"(per node)",12}");
        Console.WriteLine(new string('-', 74));
        foreach (var c in cfgs)
        {
            var m = new AlgFormer(c.Vocab, shifts: c.S, layers: c.L, maxContext: c.Ctx, dModel: c.D, frozenPrefix: c.Frozen, embedSeed: null, seed: 1);
            long p = m.ParamCount;
            double wMB = p * 8.0 / 1e6;             // weights, double[]
            double actMB = (long)c.Ctx * c.D * c.L * 8 * 4 / 1e6;   // ~activations for one sequence (rough: ctx*d*L, a few buffers)
            double nodeMB = wMB * 2 + actMB;        // weights + gradient buffer + activations
            double gradMB = wMB;                    // a full gradient is one weight-sized payload
            Console.WriteLine($"  {c.Name,-26}{p,12:N0}{wMB,9:F1}MB{nodeMB,9:F0}MB{gradMB,10:F1}MB");
        }
        Console.WriteLine(new string('-', 74));
        Console.WriteLine("  *node RAM = weights + gradient buffer + per-sequence activations (plain SGD: no optimizer moments).");
        Console.WriteLine("   fp32 would halve weights/grad; the CPU wall is compute (train speed), not RAM.\n");

        Console.WriteLine("COORDINATOR (relay-star) sizing — the server just routes; it does not need much RAM:");
        Console.WriteLine("  holds 1 model copy + in-flight gradients from N nodes. For the lean-128 base model:");
        var baseM = new AlgFormer(95, shifts: 64, layers: 8, maxContext: 256, dModel: 128, frozenPrefix: 64, seed: 1);
        double g = baseM.ParamCount * 8.0 / 1e6;
        foreach (var n in new[] { 10, 50, 200 })
            Console.WriteLine($"    {n,3} nodes: RAM ~{g * (1 + 0.5 * n):F0}MB (model + buffers)   egress/round ~{g * n * 2:F0}MB (N grads up + N applies down)");
        Console.WriteLine("\n  => RAM is trivial (a $5 VPS is plenty). BANDWIDTH is the cost: every synced gradient flows through the star.");
        Console.WriteLine("     Mitigations already in the design: positions-not-data (done), local-SGD/FedAvg (sync rarely, not every step),");
        Console.WriteLine("     gradient compression, and tree/P2P gossip to take the relay off the critical path at scale.");
    }
}
