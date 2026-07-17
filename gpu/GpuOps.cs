// Copyright (c) 2026 Dongyang Stephen Chen, trading as Evaluated Applications. All rights reserved.
// Source-available for study and verification; commercial use, redistribution, or derivatives require permission. See LICENSE.

using System.Diagnostics;
using ILGPU;
using ILGPU.Runtime;

namespace PrismFormer.Gpu;

/// <summary>
/// AlgFormer-specific GPU kernels, verified against the CPU definition. The distinctive op is the RELATION BANK
/// (AlgFormer.AlgApply): y[i] = Σ_{k&lt;S} bank[k][i]·x[(i+k) mod d] — a circular shift-and-scale, NOT a matmul, so it
/// needs its own kernel. Batched over BT = (batch × positions) rows: one GPU thread per (row, output-index i). This is
/// three of the per-layer ops (Rq, Rk, Rv, Ro), so getting it right + fast is the backbone of the forward port.
/// </summary>
public static class GpuOps
{
    // y[row, i] = Σ_{k<S} bank[k, i] * x[row, (i+k) mod d]     (flat row-major: x[row*d+i], bank[k*d+i])
    static void RelBankKernel(Index1D idx, ArrayView<float> x, ArrayView<float> bank, ArrayView<float> y, int d, int s)
    {
        int row = idx / d, i = idx % d;
        float acc = 0f;
        for (var k = 0; k < s; k++) { var xi = i + k; if (xi >= d) xi -= d; acc += bank[k * d + i] * x[row * d + xi]; }
        y[row * d + i] = acc;
    }

    /// <summary>Verify the relation-bank kernel against the CPU double definition, and time it. Returns (worstErr, gpuMs, cpuMs).</summary>
    public static (double worst, double gpuMs, double cpuMs) VerifyRelBank(int BT = 4096, int d = 256, int s = 16, int seed = 1)
    {
        var acc = GpuDevice.Accelerator ?? throw new InvalidOperationException("no accelerator");
        var rng = new Random(seed);
        var x = new float[BT * d]; var bank = new float[s * d];
        for (var i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() - 0.5);
        for (var i = 0; i < bank.Length; i++) bank[i] = (float)(rng.NextDouble() - 0.5);

        var kernel = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int>(RelBankKernel);
        using var dx = acc.Allocate1D(x);
        using var dbank = acc.Allocate1D(bank);
        using var dy = acc.Allocate1D<float>(BT * d);
        kernel(BT * d, dx.View, dbank.View, dy.View, d, s);
        acc.Synchronize();
        var y = dy.GetAsArray1D();

        // CPU reference in DOUBLE (the definition) — measures fp32-vs-fp64 closeness AND correctness of the shift logic
        var yc = new double[BT * d];
        var sw = Stopwatch.StartNew();
        for (var row = 0; row < BT; row++)
            for (var i = 0; i < d; i++)
            {
                double a = 0; for (var k = 0; k < s; k++) { var xi = i + k; if (xi >= d) xi -= d; a += (double)bank[k * d + i] * x[row * d + xi]; }
                yc[row * d + i] = a;
            }
        var cpuMs = sw.Elapsed.TotalMilliseconds;
        double worst = 0; for (var i = 0; i < y.Length; i++) worst = Math.Max(worst, Math.Abs(y[i] - yc[i]));

        const int reps = 200;
        sw.Restart();
        for (var r = 0; r < reps; r++) kernel(BT * d, dx.View, dbank.View, dy.View, d, s);
        acc.Synchronize();
        var gpuMs = sw.Elapsed.TotalMilliseconds / reps;
        return (worst, gpuMs, cpuMs);
    }

    // Backward of relbank (mirrors AlgFormer.AlgBack). Two race-free outputs:
    //   gbank[k,i] = Σ_row dOut[row,i]·x[row,(i+k)%d]              (one thread per (k,i), reduces over rows)
    //   dX[row,p]  = Σ_k   dOut[row,(p-k)%d]·bank[k,(p-k)%d]       (one thread per (row,p), reduces over k)
    static void GBankKernel(Index1D idx, ArrayView<float> dOut, ArrayView<float> x, ArrayView<float> gbank, int BT, int d, int s)
    { int i = idx % d, k = idx / d; float a = 0f; for (var row = 0; row < BT; row++) { var xi = i + k; if (xi >= d) xi -= d; a += dOut[row * d + i] * x[row * d + xi]; } gbank[idx] = a; }
    static void DxKernel(Index1D idx, ArrayView<float> dOut, ArrayView<float> bank, ArrayView<float> dX, int d, int s)
    { int p = idx % d, row = idx / d; float a = 0f; for (var k = 0; k < s; k++) { var q = p - k; if (q < 0) q += d; a += dOut[row * d + q] * bank[k * d + q]; } dX[idx] = a; }

    /// <summary>Verify the relation-bank BACKWARD (gbank + dX) against the CPU double definition of AlgBack.</summary>
    public static (double worstG, double worstX) VerifyRelBankBackward(int BT = 512, int d = 256, int s = 16, int seed = 2)
    {
        var acc = GpuDevice.Accelerator ?? throw new InvalidOperationException("no accelerator");
        var rng = new Random(seed);
        var x = new float[BT * d]; var bank = new float[s * d]; var dOut = new float[BT * d];
        for (var i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() - 0.5);
        for (var i = 0; i < bank.Length; i++) bank[i] = (float)(rng.NextDouble() - 0.5);
        for (var i = 0; i < dOut.Length; i++) dOut[i] = (float)(rng.NextDouble() - 0.5);

        var gb = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int>(GBankKernel);
        var dxk = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int>(DxKernel);
        using var dx = acc.Allocate1D(x); using var dbank = acc.Allocate1D(bank); using var ddout = acc.Allocate1D(dOut);
        using var dgbank = acc.Allocate1D<float>(s * d); using var ddx = acc.Allocate1D<float>(BT * d);
        gb(s * d, ddout.View, dx.View, dgbank.View, BT, d, s);
        dxk(BT * d, ddout.View, dbank.View, ddx.View, d, s);
        acc.Synchronize();
        var gbGpu = dgbank.GetAsArray1D(); var dxGpu = ddx.GetAsArray1D();

        // CPU reference in double, straight from the AlgBack formula
        var gbCpu = new double[s * d]; var dxCpu = new double[BT * d];
        for (var row = 0; row < BT; row++)
            for (var i = 0; i < d; i++)
                for (var k = 0; k < s; k++) { var xi = i + k; if (xi >= d) xi -= d; gbCpu[k * d + i] += (double)dOut[row * d + i] * x[row * d + xi]; dxCpu[row * d + xi] += (double)dOut[row * d + i] * bank[k * d + i]; }

        double wg = 0, wx = 0;
        for (var i = 0; i < gbCpu.Length; i++) wg = Math.Max(wg, Math.Abs(gbGpu[i] - gbCpu[i]));
        for (var i = 0; i < dxCpu.Length; i++) wx = Math.Max(wx, Math.Abs(dxGpu[i] - dxCpu[i]));
        return (wg, wx);
    }
}
