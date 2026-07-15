using System.Threading.Tasks;
using PrismFormer;

// PrismFormer reproducibility kit. Run from the repo root:  dotnet run --project verify/Verify -c Release
// Two decisive, instant checks (no training). The head-to-head generalisation numbers are the `bench` project.

Console.WriteLine("PrismFormer — verify the claims yourself (no training, runs in seconds)\n");

int pass = 0, total = 0;
void Check(string name, bool ok, string detail)
{
    total++; if (ok) pass++;
    Console.WriteLine($"  [{(ok ? "PASS" : "FAIL")}] {name}  —  {detail}");
}

// ── Claim 1: arithmetic is a property of the phasor codec, with NO training and no calculator. ──
// Encoding two numbers as phasor faces and BINDING them (one per-component complex multiply) adds
// their values on the linear-phase band and multiplies them on the log-phase band; the result is
// read back by correlation. No weights, no lookup table — just the codec.
Console.WriteLine("Claim 1 — arithmetic emerges from the representation (no training, no calculator):");
{
    var demo = PhasorCodec.Bind(PhasorCodec.NumberFace(6), PhasorCodec.NumberFace(9));
    Console.WriteLine($"    bind(face(6), face(9))  ->  sum = {PhasorCodec.DecodeSum(demo, 18)} (want 15),  product = {PhasorCodec.DecodeProduct(demo, 81)} (want 54)");

    int sumOk = 0, sumN = 0, mulOk = 0, mulN = 0;
    for (int a = 0; a <= 9; a++)
        for (int b = 0; b <= 9; b++)
        {
            var bound = PhasorCodec.Bind(PhasorCodec.NumberFace(a), PhasorCodec.NumberFace(b));
            sumN++;
            if (PhasorCodec.DecodeSum(bound, 18) == a + b) sumOk++;
            if (a >= 1 && b >= 1) { mulN++; if (PhasorCodec.DecodeProduct(bound, 81) == a * b) mulOk++; }
        }
    Check("addition by binding, operands 0..9", sumOk == sumN, $"{sumOk}/{sumN} exact");
    Check("multiplication by binding, operands 1..9", mulOk == mulN, $"{mulOk}/{mulN} exact");
}

// ── Claim 2: training is EXACTLY parallelisable — the property that enables the swarm. ──
// A batch is split into shards. Computing those shards SEQUENTIALLY and IN PARALLEL, then reducing
// in the same fixed order, gives gradients that are bit-for-bit identical (an FNV checksum over every
// gradient value). Because the substrate is exact IEEE double, a slow machine's contribution and a
// fast one's combine into the same bits — so a swarm of mismatched CPUs stays mathematically coherent.
Console.WriteLine();
Console.WriteLine("Claim 2 — sharded gradients are bit-for-bit identical whether computed serially or in parallel:");
{
    var model = AlgFormer.Mini(vocab: 256);
    var rng = new Random(1);
    var batch = new (int[] ctx, int tgt)[64];
    for (int i = 0; i < batch.Length; i++)
    {
        var ctx = new int[10];
        for (int j = 0; j < ctx.Length; j++) ctx[j] = rng.Next(256);
        batch[i] = (ctx, rng.Next(256));
    }
    const int K = 4;

    ulong ShardMerge(bool parallel)
    {
        var shards = new AlgFormer.Grads[K];
        for (int k = 0; k < K; k++) shards[k] = model.NewGrads();
        void Work(int k) { for (int i = 0; i < batch.Length; i++) if (i % K == k) model.Accumulate(batch[i].ctx, batch[i].tgt, shards[k]); }
        if (parallel) Parallel.For(0, K, Work); else for (int k = 0; k < K; k++) Work(k);
        var merged = model.NewGrads();
        for (int k = 0; k < K; k++) merged.Add(shards[k]);   // fixed index-order reduction
        return model.GradSignature(merged);
    }

    var seq  = ShardMerge(parallel: false);
    var par  = ShardMerge(parallel: true);
    var seq2 = ShardMerge(parallel: false);
    Console.WriteLine($"    sequential  FNV = 0x{seq:X16}");
    Console.WriteLine($"    parallel    FNV = 0x{par:X16}");
    Check("parallel == sequential, bitwise", seq == par, seq == par ? "identical" : "DIFFER");
    Check("fully deterministic (re-run matches)", seq == seq2, seq == seq2 ? "identical" : "DIFFER");
}

Console.WriteLine($"\n{pass}/{total} checks passed.");
Console.WriteLine();
Console.WriteLine("For the head-to-head numbers — PrismFormer vs a parameter-matched dense transformer,");
Console.WriteLine("each a mean over seeds — run the benchmarks from the repo root:");
Console.WriteLine("  dotnet run --project bench -c Release                # comparison + relational (train vs held-out)");
Console.WriteLine("  dotnet run --project bench -c Release -- --columnar  # worked multi-digit arithmetic + frozen-identity ablation");
Console.WriteLine("  dotnet run --project bench -c Release -- --lm        # next-character language modelling");
Console.WriteLine("  dotnet run --project bench -c Release -- --inspect   # mechanistic: reading the answer from the faces");

return pass == total ? 0 : 1;
