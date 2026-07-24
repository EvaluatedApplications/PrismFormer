// Copyright (c) 2026 Dongyang Stephen Chen, trading as Evaluated Applications. All rights reserved.
// Source-available for study and verification; commercial use, redistribution, or derivatives require permission. See LICENSE.

using System.Diagnostics;
using ILGPU;
using ILGPU.Runtime;

namespace PrismFormer.Gpu;

/// <summary>
/// `prismformer-gpu prec` — RELATION-BANK KERNEL VARIANTS. Training is compute-bound (CUDA pinned, VRAM slack) and the op
/// runs far below FLOP-peak, so it's memory/reuse-bound, not arithmetic-bound. The core op is
/// <c>y[i] = Σ_k bank[k][i]·x[(i+k) mod d]</c>. Three variants are timed at production width so we only port a measured win:
///   modulo         — naive %, the slow way (baseline for how bad % is),
///   cond-subtract  — the kernel SHIPPING today (i+k, subtract d on overflow — already modulo-free),
///   shared-mem tile— one thread-block per position, load x[d] into shared memory once, reuse across the d outputs
///                    (x is re-read S times per position from global otherwise). This is the real memory lever.
/// (fp16 was tested earlier and is a wash on ILGPU — scalar half, no packed/tensor-core ops — so it's dropped here.)
/// </summary>
public static class GpuComputeBench
{
    public static void Run()
    {
        var acc = GpuDevice.Accelerator;
        if (acc == null) { Console.WriteLine("no accelerator"); return; }
        Console.WriteLine($"relation-bank kernel variants — {GpuDevice.Describe}");
        Console.WriteLine("  y[i]=Σ_k bank[k][i]·x[(i+k)%d]   d256 S64, B64 T128, fp32\n");

        const int d = 256, S = 64, B = 64, T = 128, iters = 400;
        int N = B * T * d, rows = B * T;
        double flops = 2.0 * S * (double)N * iters;
        using var x = acc.Allocate1D<float>(N);
        using var bank = acc.Allocate1D<float>(S * d);
        using var y = acc.Allocate1D<float>(N);

        // 1. naive modulo
        var kMod = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int>(
            (idx, xx, bb, yy, dd, s) => { int gi = idx; int i = gi % dd; int b = (gi / dd) * dd; float a = 0f; for (int k = 0; k < s; k++) a += bb[k * dd + i] * xx[b + ((i + k) % dd)]; yy[idx] = a; });
        double msMod = Time(acc, iters, () => kMod(N, x.View, bank.View, y.View, d, S));

        // 2. conditional-subtract — the kernel running today
        var kCond = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int>(
            (idx, xx, bb, yy, dd, s) => { int row = idx / dd, i = idx % dd; float a = 0f; for (int k = 0; k < s; k++) { int xi = i + k; if (xi >= dd) xi -= dd; a += bb[k * dd + i] * xx[row * dd + xi]; } yy[idx] = a; });
        double msCond = Time(acc, iters, () => kCond(N, x.View, bank.View, y.View, d, S));

        // 3. shared-memory x-tile — one block per position (d threads), stage x[d] in shared, reuse across the d outputs
        var kSmem = acc.LoadStreamKernel<ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int>(
            (xx, bb, yy, dd, s) =>
            {
                int i = Group.IdxX, row = Grid.IdxX;
                var xs = SharedMemory.Allocate1D<float>(256);   // d = 256 (production width)
                xs[i] = xx[row * dd + i];
                Group.Barrier();
                float a = 0f;
                for (int k = 0; k < s; k++) { int xi = i + k; if (xi >= dd) xi -= dd; a += bb[k * dd + i] * xs[xi]; }
                yy[row * dd + i] = a;
            });
        var cfg = new KernelConfig(rows, d);
        double msSmem = Time(acc, iters, () => kSmem(cfg, x.View, bank.View, y.View, d, S));

        Console.WriteLine($"  {"modulo (naive)",-26}{msMod,8:F1} ms{flops / msMod / 1e6,9:F0} GFLOP/s   {msCond / msMod,5:F2}x vs current");
        Console.WriteLine($"  {"cond-subtract (current)",-26}{msCond,8:F1} ms{flops / msCond / 1e6,9:F0} GFLOP/s    1.00x");
        Console.WriteLine($"  {"shared-mem x-tile",-26}{msSmem,8:F1} ms{flops / msSmem / 1e6,9:F0} GFLOP/s   {msCond / msSmem,5:F2}x vs current");
        Console.WriteLine("\n  read: shared-mem >1.2x vs current = a real win, worth porting the relbank kernel. ~1x = the current");
        Console.WriteLine("  kernel is already near its memory-bound ceiling and the compute genuinely is what it is.");

        // correctness spot-check: shared-mem must match cond-subtract
        var r = new Random(1); var hx = new float[N]; var hb = new float[S * d];
        for (var j = 0; j < N; j++) hx[j] = (float)(r.NextDouble() * 2 - 1);
        for (var j = 0; j < S * d; j++) hb[j] = (float)(r.NextDouble() * 2 - 1);
        x.CopyFromCPU(hx); bank.CopyFromCPU(hb);
        kCond(N, x.View, bank.View, y.View, d, S); acc.Synchronize(); var yc = y.GetAsArray1D();
        kSmem(cfg, x.View, bank.View, y.View, d, S); acc.Synchronize(); var ys = y.GetAsArray1D();
        double maxDiff = 0; for (var j = 0; j < N; j++) maxDiff = Math.Max(maxDiff, Math.Abs(yc[j] - ys[j]));
        Console.WriteLine($"  correctness: max|cond - smem| = {maxDiff:E2}  ({(maxDiff < 1e-3 ? "MATCH" : "MISMATCH")})");
    }

    static double Time(Accelerator acc, int iters, Action launch)
    {
        launch(); acc.Synchronize();   // warm (compile + first run)
        var sw = Stopwatch.StartNew();
        for (var i = 0; i < iters; i++) launch();
        acc.Synchronize();
        return sw.Elapsed.TotalMilliseconds;
    }
}
