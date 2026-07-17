// Copyright (c) 2026 Dongyang Stephen Chen, trading as Evaluated Applications. All rights reserved.
// Source-available for study and verification; commercial use, redistribution, or derivatives require permission. See LICENSE.

using System.Diagnostics;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.CPU;

// PoC: prove the ILGPU -> CUDA toolchain works end-to-end on this machine before investing in the AlgFormer port.
//   prismformer-gpu            -> vector-add correctness + a throughput sanity check on the GPU

var mode = args.Length > 0 ? args[0] : "poc";

if (mode == "poc")
{
    Console.WriteLine("PrismFormerGpu PoC — ILGPU/CUDA toolchain check\n");

    using var context = Context.Create(b => b.Cuda().CPU());   // CUDA if present; CPU accelerator as a fallback so this never hard-fails
    Console.WriteLine("devices:");
    foreach (var d in context.Devices) Console.WriteLine($"  {d.AcceleratorType,-6} {d.Name}");

    var device = context.GetPreferredDevice(preferCPU: false);
    using var acc = device.CreateAccelerator(context);
    Console.WriteLine($"\nusing: {acc.AcceleratorType}  {acc.Name}  ({acc.MemorySize / (1024 * 1024)} MB)\n");

    const int N = 1 << 22;   // 4M floats
    var a = new float[N]; var b = new float[N];
    for (var i = 0; i < N; i++) { a[i] = i; b[i] = 2f * i; }

    var kernel = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(
        (i, x, y, z) => z[i] = x[i] + y[i]);

    using var da = acc.Allocate1D(a);
    using var db = acc.Allocate1D(b);
    using var dc = acc.Allocate1D<float>(N);

    kernel((int)da.Length, da.View, db.View, dc.View);
    acc.Synchronize();
    var c = dc.GetAsArray1D();

    var ok = true; double worst = 0;
    for (var i = 0; i < N; i++) { var e = Math.Abs(c[i] - (a[i] + b[i])); if (e > worst) worst = e; if (e > 1e-2) ok = false; }
    Console.WriteLine($"vector-add ({N:N0} elems): {(ok ? "PASSED" : "FAILED")}  (worst abs err {worst:E2})");

    // rough throughput: repeat the add many times, measure GPU vs a single-thread CPU loop
    const int reps = 200;
    var sw = Stopwatch.StartNew();
    for (var r = 0; r < reps; r++) { kernel((int)da.Length, da.View, db.View, dc.View); }
    acc.Synchronize();
    var gpuMs = sw.Elapsed.TotalMilliseconds / reps;

    sw.Restart();
    for (var r = 0; r < reps; r++) for (var i = 0; i < N; i++) c[i] = a[i] + b[i];
    var cpuMs = sw.Elapsed.TotalMilliseconds / reps;

    Console.WriteLine($"throughput/pass: GPU {gpuMs:F2} ms   CPU(1 thread) {cpuMs:F2} ms   speedup ~{cpuMs / Math.Max(0.001, gpuMs):F1}x");
    Console.WriteLine("\nIf this used a Cuda accelerator and PASSED, the toolchain is live — the AlgFormer kernel port is unblocked.");
}
else if (mode == "matmul")
{
    // The op that dominates AlgFormer training: a batched matmul Y[B,M] = X[B,K] @ W[K,M]. Compute-bound, so this is
    // where the real training speedup lives (unlike the memory-bound vector-add). Correctness vs CPU + timing.
    using var context = Context.Create(b => b.Cuda().CPU());
    var device = context.GetPreferredDevice(preferCPU: false);
    using var acc = device.CreateAccelerator(context);
    Console.WriteLine($"batched matmul on {acc.AcceleratorType} {acc.Name}\n");

    const int B = 4096, K = 256, M = 256;   // batch x (dim -> dim), ~268M MACs — an FFN-sized op
    var rng = new Random(1);
    var x = new float[B * K]; var w = new float[K * M];
    for (var i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() - 0.5);
    for (var i = 0; i < w.Length; i++) w[i] = (float)(rng.NextDouble() - 0.5);

    var mm = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int>(
        (i, xv, wv, yv, b, k, m) => { var row = i / m; var col = i % m; float a = 0; for (var kk = 0; kk < k; kk++) a += xv[row * k + kk] * wv[kk * m + col]; yv[i] = a; });

    using var dx = acc.Allocate1D(x);
    using var dw = acc.Allocate1D(w);
    using var dy = acc.Allocate1D<float>(B * M);

    mm(B * M, dx.View, dw.View, dy.View, B, K, M);
    acc.Synchronize();
    var y = dy.GetAsArray1D();

    // CPU reference (single thread) + correctness
    var yc = new float[B * M];
    var swc = Stopwatch.StartNew();
    for (var b = 0; b < B; b++) for (var m = 0; m < M; m++) { float a = 0; for (var k = 0; k < K; k++) a += x[b * K + k] * w[k * M + m]; yc[b * M + m] = a; }
    var cpuMs = swc.Elapsed.TotalMilliseconds;
    double worst = 0; for (var i = 0; i < y.Length; i++) worst = Math.Max(worst, Math.Abs(y[i] - yc[i]));
    Console.WriteLine($"correctness vs CPU: worst abs err {worst:E2}  ({(worst < 1e-3 ? "PASS (fp32-close)" : "CHECK")})");

    const int reps = 100;
    var sw = Stopwatch.StartNew();
    for (var r = 0; r < reps; r++) mm(B * M, dx.View, dw.View, dy.View, B, K, M);
    acc.Synchronize();
    var gpuMs = sw.Elapsed.TotalMilliseconds / reps;
    Console.WriteLine($"per matmul ({B}x{K}x{M}): GPU {gpuMs:F3} ms   CPU(1 thread) {cpuMs:F1} ms   speedup ~{cpuMs / Math.Max(0.001, gpuMs):F0}x");
    Console.WriteLine("\nThis is the compute-bound path that dominates AlgFormer. A big speedup here = the port is worth it.");
}
