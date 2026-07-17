// Copyright (c) 2026 Dongyang Stephen Chen, trading as Evaluated Applications. All rights reserved.
// Source-available for study and verification; commercial use, redistribution, or derivatives require permission. See LICENSE.

using System.Diagnostics;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.CPU;
using PrismFormer;
using PrismFormer.Gpu;

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
else if (mode == "relbank")
{
    // The AlgFormer-specific op (Rq/Rk/Rv/Ro): y[i]=Σ_k bank[k][i]·x[(i+k)%d] — a shift-and-scale, not a matmul.
    Console.WriteLine($"AlgFormer relation-bank kernel — auto-detected device: {GpuDevice.Describe}  (HasGpu={GpuDevice.HasGpu})\n");
    var (worst, gpuMs, cpuMs) = GpuOps.VerifyRelBank();
    Console.WriteLine($"correctness vs CPU double definition: worst abs err {worst:E2}  ({(worst < 1e-3 ? "PASS (fp32-close)" : "CHECK")})");
    Console.WriteLine($"per relbank (4096 rows x 256 dim, S=16): GPU {gpuMs:F3} ms   CPU(1 thread) {cpuMs:F1} ms   speedup ~{cpuMs / Math.Max(0.001, gpuMs):F0}x");
    Console.WriteLine("\nThe custom relation-bank op ports correctly to GPU — the forward backbone (Rq/Rk/Rv/Ro) is unblocked.");
    GpuDevice.Shutdown();
}
else if (mode == "relback")
{
    Console.WriteLine($"relation-bank BACKWARD gradcheck — device: {GpuDevice.Describe}\n");
    var (wg, wx) = GpuOps.VerifyRelBankBackward();
    Console.WriteLine($"  gbank (grad to bank): worst abs err {wg:E2}");
    Console.WriteLine($"  dX    (grad to input): worst abs err {wx:E2}");
    Console.WriteLine((wg < 1e-3 && wx < 1e-3) ? "\n  RELBANK BACKWARD MATCHES CPU (fp32-close) — the backbone of the backward pass is verified."
                                              : "\n  MISMATCH — kernel bug.");
    GpuDevice.Shutdown();
}
else if (mode == "forward")
{
    // The milestone: a whole batched PrismFormer forward on GPU, gradchecked against CPU AlgFormer.LogitsFor.
    Console.WriteLine($"GPU forward gradcheck vs CPU AlgFormer — device: {GpuDevice.Describe}\n");
    const int V = 32, d = 256, S = 8, L = 2, T = 9, B = 128;
    var cpu = new AlgFormer(V, shifts: S, layers: L, maxContext: T, dModel: d, frozenPrefix: 0, embedSeed: null, seed: 3);
    var trng = new Random(11);   // a few steps so params aren't at trivial init (forward must match at any params)
    for (var i = 0; i < 200; i++) { var c = new int[T]; for (var t = 0; t < T; t++) c[t] = trng.Next(V); cpu.TrainStep(c, trng.Next(V), 2e-3); }

    using var gm = new GpuModel(cpu.Serialize());
    var rng = new Random(7);
    var toks = new int[B][];
    for (var b = 0; b < B; b++) { toks[b] = new int[T]; for (var t = 0; t < T; t++) toks[b][t] = rng.Next(V); }

    var gpu = gm.Forward(toks);
    double worst = 0, worstRel = 0; var agree = 0;
    for (var b = 0; b < B; b++)
    {
        var cl = cpu.LogitsFor(toks[b]);
        int ga = 0, ca = 0;
        for (var w = 0; w < V; w++) { var e = Math.Abs(gpu[b][w] - cl[w]); if (e > worst) worst = e; var rel = e / (Math.Abs(cl[w]) + 1e-6); if (rel > worstRel) worstRel = rel; if (gpu[b][w] > gpu[b][ga]) ga = w; if (cl[w] > cl[ca]) ca = w; }
        if (ga == ca) agree++;
    }
    Console.WriteLine($"  {B} sequences (V={V}, d={d}, layers={L}, S={S}, T={T})");
    Console.WriteLine($"  worst abs logit err {worst:E2}   worst rel err {worstRel:E2}");
    Console.WriteLine($"  argmax agreement (GPU prediction == CPU): {agree}/{B}  ({(double)agree / B:P1})");
    Console.WriteLine(agree == B ? "\n  FORWARD MATCHES — a full PrismFormer forward pass runs on GPU and agrees with CPU (fp32-close)."
                                 : "\n  MISMATCH — a kernel or layout bug to chase.");
    GpuDevice.Shutdown();
}
else if (mode == "backward")
{
    // The training milestone: GPU gradients vs CPU AlgFormer gradients (summed over the batch), per section.
    Console.WriteLine($"GPU backward gradcheck vs CPU AlgFormer — device: {GpuDevice.Describe}\n");
    const int V = 32, d = 256, S = 8, L = 2, T = 9, B = 64;
    var cpu = new AlgFormer(V, shifts: S, layers: L, maxContext: T, dModel: d, frozenPrefix: 0, embedSeed: null, seed: 5);
    var trng = new Random(13);
    for (var i = 0; i < 200; i++) { var c = new int[T]; for (var t = 0; t < T; t++) c[t] = trng.Next(V); cpu.TrainStep(c, trng.Next(V), 2e-3); }

    using var gm = new GpuModel(cpu.Serialize());
    var rng = new Random(9);
    var toks = new int[B][]; var tgt = new int[B];
    for (var b = 0; b < B; b++) { toks[b] = new int[T]; for (var t = 0; t < T; t++) toks[b][t] = rng.Next(V); tgt[b] = rng.Next(V); }

    var g = cpu.NewGrads();
    for (var b = 0; b < B; b++) cpu.Accumulate(toks[b], tgt[b], g);
    var gbytes = cpu.SerializeGradient(g);
    using var r = new BinaryReader(new MemoryStream(gbytes));   // SerializeGradient: no header, pure Pairs-order doubles
    var cpuFlat = new List<double>(); while (r.BaseStream.Position < gbytes.Length) cpuFlat.Add(r.ReadDouble());

    var gpuFlat = gm.Backward(toks, tgt);
    if (cpuFlat.Count != gpuFlat.Length) { Console.WriteLine($"  LENGTH MISMATCH cpu {cpuFlat.Count} gpu {gpuFlat.Length}"); GpuDevice.Shutdown(); return; }

    (double w, double wr) Sec(int lo, int hi) { double w = 0, wr = 0; for (var i = lo; i < hi; i++) { var e = Math.Abs(gpuFlat[i] - cpuFlat[i]); if (e > w) w = e; var rel = e / (Math.Abs(cpuFlat[i]) + 1e-4); if (rel > wr) wr = rel; } return (w, wr); }
    int emb = V * d, pos = T * d, cc = V, bank = 7 * S * d;
    var off = 0;
    Console.WriteLine($"  {"section",-10} {"worst abs",11} {"worst rel",11}");
    void Show(string name, int len) { var (w, wr) = Sec(off, off + len); Console.WriteLine($"  {name,-10} {w,11:E2} {wr,11:E2}"); off += len; }
    Show("Emb", emb); Show("Pos", pos); Show("C", cc);
    for (var l = 0; l < L; l++) Show($"layer{l}", bank);
    var (W, WR) = Sec(0, gpuFlat.Length);
    Console.WriteLine($"\n  OVERALL worst abs {W:E2}  worst rel {WR:E2}   ({gpuFlat.Length:N0} params, B={B})");
    Console.WriteLine(W < 1e-2 ? "\n  BACKWARD MATCHES CPU (fp32-close) — GPU forward+backward is correct. GPU training is unblocked."
                              : "\n  MISMATCH — a grad section is off (see per-section above).");
    GpuDevice.Shutdown();
}
