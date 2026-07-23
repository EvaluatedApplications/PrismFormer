// Copyright (c) 2026 Dongyang Stephen Chen, trading as Evaluated Applications. All rights reserved.
// Source-available for study and verification; commercial use, redistribution, or derivatives require permission. See LICENSE.

using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.CPU;
using ILGPU.Algorithms;

namespace PrismFormer.Gpu;

/// <summary>
/// AUTO-DETECT the compute device once, cached process-wide: a CUDA GPU if one is present, otherwise the CPU
/// accelerator as a safe fallback. Everything GPU-side routes through <see cref="Accelerator"/>, so the same code
/// runs on either — a box with an NVIDIA GPU uses it, a box without one still works (just slower). Lazy + thread-safe:
/// the driver probe happens on first use and never again. Dispose at process exit via <see cref="Shutdown"/>.
/// </summary>
public static class GpuDevice
{
    static readonly object _gate = new();
    static Context? _ctx;
    static Accelerator? _acc;
    static bool _init;

    static void Init()
    {
        if (_init) return;
        lock (_gate)
        {
            if (_init) return;
            try
            {
                _ctx = Context.Create(b => b.Cuda().CPU().EnableAlgorithms());
                var cuda = _ctx.Devices.FirstOrDefault(d => d.AcceleratorType == AcceleratorType.Cuda);
                _acc = cuda is not null ? cuda.CreateAccelerator(_ctx) : _ctx.GetPreferredDevice(preferCPU: true).CreateAccelerator(_ctx);
            }
            catch { _ctx = null; _acc = null; }   // no ILGPU/driver at all → callers fall back to the pure-CPU AlgFormer
            _init = true;
        }
    }

    /// <summary>True iff a real CUDA GPU was found (not the CPU fallback). Cheap after first call.</summary>
    public static bool HasGpu { get { Init(); return _acc is not null && _acc.AcceleratorType == AcceleratorType.Cuda; } }

    /// <summary>The chosen accelerator (CUDA GPU or CPU fallback), or null if ILGPU/driver is unavailable entirely.</summary>
    public static Accelerator? Accelerator { get { Init(); return _acc; } }

    /// <summary>Human-readable device line for logs — "NVIDIA RTX A3000 …" or "CPU (no CUDA GPU)".</summary>
    public static string Describe { get { Init(); return _acc is null ? "none (ILGPU unavailable)" : _acc.AcceleratorType == AcceleratorType.Cuda ? $"GPU: {_acc.Name} ({_acc.MemorySize / (1024 * 1024)} MB)" : "CPU (no CUDA GPU)"; } }

    /// <summary>Total device memory in MB, for sizing the GPU sub-batch to the card. 0 if no accelerator.</summary>
    public static long MemoryMb { get { Init(); return _acc is null ? 0 : _acc.MemorySize / (1024 * 1024); } }

    public static void Shutdown()
    {
        lock (_gate) { try { _acc?.Dispose(); _ctx?.Dispose(); } catch { } _acc = null; _ctx = null; _init = false; }
    }
}
