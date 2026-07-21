// Copyright (c) 2026 Dongyang Stephen Chen, trading as Evaluated Applications. All rights reserved.
// Source-available for study and verification; commercial use, redistribution, or derivatives require permission. See LICENSE.

using System.Numerics;

namespace PrismFormer;

/// <summary>
/// AlgFormer — the algebraic transformer, Prism's first-class learner. It is the transformer shape (multi-head
/// attention → per-token feed-forward → residuals, stacked), but every dense linear map W·x is the validated
/// algebraic cell — a relation-bank of FACE parameters composed by bind + permute + bundle:
///   y[i] = Σ_{k&lt;S} bank[k][i] · x[(i+k) mod d]        (bundle over bind(x)∘permute_k)
/// A bank is S·d parameters instead of d²; at S=d it exactly reparametrises a full matrix, so S is a lean↔full knob.
/// Every parameter is a face (decodable), and data and parameters live in one space under one algebra.
///
/// The face is used in FULL: embeddings carry the whole phasor face, and the identity comps (linear+log phase,
/// dims [0, FrozenPrefix)) are FROZEN — so numbers keep their exact, homomorphic value and only the orbital meaning
/// learns. Readout is tied to the face table. Gradients accumulate into a detached <see cref="Grads"/> buffer so a
/// batch splits across cores and reduces with one <see cref="Step"/> (see <see cref="Train"/>). Gradchecked.
/// </summary>
public sealed class AlgFormer
{
    private readonly int _v, _d, _layers, _maxT, _s, _frozen;
    private readonly bool _bind;   // FFN combine: false = gated GLU aval⊙σ(gt); true = complex-pair multiply aval⊛gt (the phasor bind)

    /// <summary>How the per-position forward loops fan out. Default = sequential (keeps this lib EvalApp-free and is the
    /// TRAINING path, which is already batch-parallel — must not nest). Serving sets this to the EvalApp CPU-gated map
    /// (<c>PrismEval.Cpu</c>) so a single generate uses the cores. Chunks write disjoint positions → parallel == serial.</summary>
    public IParallelMap Map { get; set; } = SequentialMap.Instance;

    internal sealed class Layer { public required double[][] Rq, Rk, Rv, Ro, A1, Ag, Ao; }

    internal readonly double[][] Emb, Pos;
    internal readonly double[] C;
    internal readonly Layer[] Ls;
    private readonly Dictionary<double[], (double[] m, double[] v)> _mom = new();
    private long _t;

    /// <summary>Full-face model with the identity codec frozen. <paramref name="dModel"/> defaults to the phasor face
    /// dimension; <paramref name="frozenPrefix"/> defaults to the frozen identity comps (linear+log phase) so a number's
    /// value stays exact. Pass dModel smaller / frozenPrefix 0 for an orbital-only (all-learned) model.</summary>
    public AlgFormer(int vocab, int shifts, int layers, int maxContext, int dModel = 0, int frozenPrefix = -1, Func<int, double[]>? embedSeed = null, int seed = 42, bool bindFfn = false)
    {
        _v = vocab; _d = dModel > 0 ? dModel : PhasorLayout.Dim; _s = Math.Min(shifts, _d); _layers = layers; _maxT = maxContext;
        _frozen = frozenPrefix >= 0 ? Math.Min(frozenPrefix, _d) : Math.Min(PhasorLayout.FrozenReals, _d);
        _bind = bindFfn;
        var rng = new Random(seed);
        double[] Row(int n, bool zero = false) { var r = new double[n]; if (!zero) for (var i = 0; i < n; i++) r[i] = (rng.NextDouble() - 0.5) * 2 * 0.08; return r; }
        double[][] Bank(bool zero = false) => Enumerable.Range(0, _s).Select(_ => Row(_d, zero)).ToArray();

        Emb = new double[_v][];
        for (var w = 0; w < _v; w++) Emb[w] = embedSeed?.Invoke(w) ?? Row(_d);
        Pos = Enumerable.Range(0, _maxT).Select(_ => Row(_d)).ToArray(); C = Row(_v, true);
        Ls = Enumerable.Range(0, layers).Select(_ => new Layer { Rq = Bank(), Rk = Bank(), Rv = Bank(), Ro = Bank(), A1 = Bank(), Ag = Bank(), Ao = Bank() }).ToArray();
    }

    /// <summary>The mini-LLM preset on the uniform phasor face (see <see cref="PhasorCodec"/>): full Dim, the identity
    /// comps (linear+log phase) frozen, 4 layers, 32 shifts, 16-token context — a real, trainable little language model
    /// whose numbers are discriminable so the tied readout can name a computed value. Seed with <see cref="PhasorCodec.Encode"/>.</summary>
    public static AlgFormer Mini(int vocab, Func<int, double[]>? embedSeed = null, int seed = 42)
        => new(vocab, shifts: 32, layers: 4, maxContext: 16, dModel: PhasorLayout.Dim, frozenPrefix: PhasorLayout.FrozenReals, embedSeed: embedSeed, seed: seed);

    public int Vocab => _v;
    public int Dim => _d;
    public int Context => _maxT;
    public int Shifts => _s;
    public int Layers => _layers;
    public long ParamCount => Pairs(NewGrads()).Sum(p => (long)p.param.Length);

    // ---- gradient buffer (detached, mergeable) ----
    public sealed class Grads
    {
        internal sealed class LayerG { public required double[][] Rq, Rk, Rv, Ro, A1, Ag, Ao; }
        internal double[][] Emb = default!, Pos = default!;
        internal double[] C = default!;
        internal LayerG[] Ls = default!;
        public void Add(Grads o)
        {
            for (var i = 0; i < Emb.Length; i++) AddV(Emb[i], o.Emb[i]);
            for (var i = 0; i < Pos.Length; i++) AddV(Pos[i], o.Pos[i]);
            AddV(C, o.C);
            for (var l = 0; l < Ls.Length; l++)
                foreach (var (a, b) in new[] { (Ls[l].Rq, o.Ls[l].Rq), (Ls[l].Rk, o.Ls[l].Rk), (Ls[l].Rv, o.Ls[l].Rv), (Ls[l].Ro, o.Ls[l].Ro), (Ls[l].A1, o.Ls[l].A1), (Ls[l].Ag, o.Ls[l].Ag), (Ls[l].Ao, o.Ls[l].Ao) })
                    for (var k = 0; k < a.Length; k++) AddV(a[k], b[k]);
        }
        private static void AddV(double[] a, double[] b) { for (var i = 0; i < a.Length; i++) a[i] += b[i]; }
        /// <summary>Zero the buffer for reuse across steps (avoids reallocating a large Emb grad every example).</summary>
        public void Clear()
        {
            foreach (var r in Emb) Array.Clear(r);
            foreach (var r in Pos) Array.Clear(r);
            Array.Clear(C);
            foreach (var l in Ls) foreach (var m in new[] { l.Rq, l.Rk, l.Rv, l.Ro, l.A1, l.Ag, l.Ao }) foreach (var r in m) Array.Clear(r);
        }
    }

    /// <summary>Seed a token's embedding from its codec face — identity bands (frozen) hold the exact value, the
    /// orbital is init. Call when a symbol is first assigned an id.</summary>
    public void Seed(int id, double[] face)
    {
        if (id < 0 || id >= _v) return;
        Array.Copy(face, Emb[id], Math.Min(face.Length, _d));
    }

    /// <summary>Restore ONLY the frozen identity prefix (first <c>frozenPrefix</c> reals) of every embedding row to its
    /// canonical codec face. Call after a weight-slice bleed / elastic average: the frozen bands hold a number's EXACT
    /// value + symbol signature and must never drift, but <c>(1-α)x + αx</c> is not bit-exactly <c>x</c> in floating point,
    /// so repeated averaging would slowly corrupt exact arithmetic. The learned orbital tail (cols ≥ frozenPrefix) is left
    /// untouched.</summary>
    public void RestoreFrozen(Func<int, double[]> face)
    {
        if (_frozen <= 0) return;
        for (var id = 0; id < _v; id++) { var f = face(id); if (f != null) Array.Copy(f, Emb[id], Math.Min(_frozen, Math.Min(f.Length, _d))); }
    }

    /// <summary>Persist parameters (not Adam moments) in Pairs order, behind a shape header.</summary>
    public void Save(System.IO.BinaryWriter w)
    {
        w.Write(_v); w.Write(_d); w.Write(_s); w.Write(_layers);
        foreach (var (p, _) in Pairs(NewGrads())) foreach (var x in p) w.Write(x);
    }

    /// <summary>Reinitialise every parameter in place from a fresh model of the SAME shape and clear the optimiser
    /// state (Adam moments + step count) — the "reset model" primitive (e.g. after a spec change orphans old weights).
    /// In place because the live model instance is shared; shapes must match (same spec).</summary>
    public void ReinitFrom(AlgFormer src)
    {
        var mine = Pairs(NewGrads()).Select(pg => pg.param).ToArray();
        var theirs = src.Pairs(src.NewGrads()).Select(pg => pg.param).ToArray();
        for (var i = 0; i < mine.Length; i++) Array.Copy(theirs[i], mine[i], mine[i].Length);
        _mom.Clear(); _t = 0;
    }

    /// <summary>Restore parameters; returns false on a shape mismatch (caller starts fresh).</summary>
    public bool Load(System.IO.BinaryReader r)
    {
        if (r.ReadInt32() != _v || r.ReadInt32() != _d || r.ReadInt32() != _s || r.ReadInt32() != _layers) return false;
        foreach (var (p, _) in Pairs(NewGrads())) for (var i = 0; i < p.Length; i++) p[i] = r.ReadDouble();
        return true;
    }

    /// <summary>UPGRADE-IN-PLACE: load an OLDER checkpoint (smaller <c>Shifts</c>, <c>Context</c>, and/or <c>Vocab</c>)
    /// into this (larger) model. New shift rows are zeroed (they contribute nothing to the algebraic sum, so output is
    /// byte-identical the instant you extend, then training fills them in); extra <c>Pos</c> rows keep their init (never
    /// indexed on ≤ old-context inputs); and VOCAB is APPEND-ONLY — the old rows [0, ov) carry over and the appended rows
    /// [ov, vocab) keep this model's construction seed (their codec faces), so the learned char knowledge is preserved and
    /// the new subword rows start at their identity. The fixed dims (dim, frozen, layers) must match. <paramref
    /// name="oldMaxT"/> is the old context length (its Pos-row count). Lets capacity grow without a retrain.</summary>
    public bool LoadUpgrade(System.IO.BinaryReader r, int oldMaxT)
    {
        int ov = r.ReadInt32(), od = r.ReadInt32(), os = r.ReadInt32(), ol = r.ReadInt32();
        if (ov > _v || od != _d || ol != _layers || os > _s || oldMaxT > _maxT) return false;   // Shifts, Context, and Vocab may GROW (vocab is APPEND-ONLY: ov <= _v)
        for (var w = 0; w < ov; w++) for (var i = 0; i < od; i++) Emb[w][i] = r.ReadDouble();     // old rows carry over; appended rows [ov.._v) keep their construction seed (codec faces)
        for (var t = 0; t < oldMaxT; t++) for (var i = 0; i < od; i++) Pos[t][i] = r.ReadDouble();   // extra Pos rows keep their init
        for (var w = 0; w < ov; w++) C[w] = r.ReadDouble();                                       // appended readout entries C[ov.._v) keep their init (0)
        for (var l = 0; l < _layers; l++)
            foreach (var bank in new[] { Ls[l].Rq, Ls[l].Rk, Ls[l].Rv, Ls[l].Ro, Ls[l].A1, Ls[l].Ag, Ls[l].Ao })
            {
                for (var k = 0; k < os; k++) for (var i = 0; i < od; i++) bank[k][i] = r.ReadDouble();
                for (var k = os; k < _s; k++) Array.Clear(bank[k]);   // ZERO the new shift rows → identity-preserving
            }
        _mom.Clear(); _t = 0;   // optimiser state is per-shape; reset (params are preserved)
        return true;
    }

    // ── distributed / federated training: gradient transport ─────────────────────────────────────────────
    /// <summary>Serialize a gradient buffer to bytes — the transport primitive for data-parallel / federated training:
    /// a worker computes its data shard's gradient, ships these bytes, and a coordinator sums the buffers
    /// (<see cref="Grads.Add"/>) and applies one <see cref="Step"/>. Layout follows <see cref="Pairs"/> order, so it
    /// round-trips exactly (lossless).</summary>
    public byte[] SerializeGradient(Grads g)
    {
        using var ms = new System.IO.MemoryStream();
        var w = new System.IO.BinaryWriter(ms);
        foreach (var (_, grad) in Pairs(g)) foreach (var x in grad) w.Write(x);
        w.Flush();
        return ms.ToArray();
    }

    /// <summary>Inverse of <see cref="SerializeGradient"/>: read a wire buffer into a fresh <see cref="Grads"/>.</summary>
    public Grads DeserializeGradient(byte[] wire)
    {
        var g = NewGrads();
        using var ms = new System.IO.MemoryStream(wire);
        using var r = new System.IO.BinaryReader(ms);
        foreach (var (_, grad) in Pairs(g)) for (var i = 0; i < grad.Length; i++) grad[i] = r.ReadDouble();
        return g;
    }

    /// <summary>Self-describing model bytes: full config (so the receiver can reconstruct the exact shape) + parameters.
    /// Lets a distributed worker rebuild any model the host sends without agreeing on a config out-of-band.</summary>
    public byte[] Serialize()
    {
        using var ms = new System.IO.MemoryStream();
        var w = new System.IO.BinaryWriter(ms);
        w.Write(_v); w.Write(_s); w.Write(_layers); w.Write(_maxT); w.Write(_d); w.Write(_frozen);
        foreach (var (p, _) in Pairs(NewGrads())) foreach (var x in p) w.Write(x);
        w.Flush(); return ms.ToArray();
    }

    /// <summary>Reconstruct a model from <see cref="Serialize"/> bytes (config + parameters).</summary>
    public static AlgFormer Deserialize(byte[] bytes)
    {
        using var ms = new System.IO.MemoryStream(bytes);
        using var r = new System.IO.BinaryReader(ms);
        int v = r.ReadInt32(), s = r.ReadInt32(), layers = r.ReadInt32(), maxT = r.ReadInt32(), d = r.ReadInt32(), frozen = r.ReadInt32();
        var m = new AlgFormer(v, s, layers, maxT, d, frozen, embedSeed: null, seed: 1);
        foreach (var (p, _) in m.Pairs(m.NewGrads())) for (var i = 0; i < p.Length; i++) p[i] = r.ReadDouble();
        return m;
    }

    public Grads NewGrads()
    {
        double[][] Z(double[][] like) => like.Select(r => new double[r.Length]).ToArray();
        return new Grads
        {
            Emb = Z(Emb), Pos = Z(Pos), C = new double[C.Length],
            Ls = Ls.Select(l => new Grads.LayerG { Rq = Z(l.Rq), Rk = Z(l.Rk), Rv = Z(l.Rv), Ro = Z(l.Ro), A1 = Z(l.A1), Ag = Z(l.Ag), Ao = Z(l.Ao) }).ToArray(),
        };
    }

    private IEnumerable<(double[] param, double[] grad)> Pairs(Grads g)
    {
        for (var i = 0; i < Emb.Length; i++) yield return (Emb[i], g.Emb[i]);
        for (var i = 0; i < Pos.Length; i++) yield return (Pos[i], g.Pos[i]);
        yield return (C, g.C);
        for (var l = 0; l < Ls.Length; l++)
            foreach (var (pm, gm) in new[] { (Ls[l].Rq, g.Ls[l].Rq), (Ls[l].Rk, g.Ls[l].Rk), (Ls[l].Rv, g.Ls[l].Rv), (Ls[l].Ro, g.Ls[l].Ro), (Ls[l].A1, g.Ls[l].A1), (Ls[l].Ag, g.Ls[l].Ag), (Ls[l].Ao, g.Ls[l].Ao) })
                for (var k = 0; k < pm.Length; k++) yield return (pm[k], gm[k]);
    }

    /// <summary>Bitwise FNV-1a checksum over EVERY gradient value (all params via <see cref="Pairs"/>) — a bit-identity
    /// guard for the backprop hot path: any change that alters a single gradient bit changes this hash. Used by bench
    /// --gradcheck to prove the scratch-reuse refactor is numerically identical to the allocating path.</summary>
    public ulong GradSignature(Grads g)
    {
        var h = 1469598103934665603UL;
        foreach (var (_, grad) in Pairs(g)) foreach (var x in grad) { h ^= BitConverter.DoubleToUInt64Bits(x); h *= 1099511628211UL; }
        return h;
    }

    // algebraic linear map:  y[i] = Σ_k bank[k][i] · x[(i+k) mod d]
    // Shift-outer + SIMD over the OUTPUT index i: for a fixed shift k, x[(i+k) mod d] is x rotated by k = two CONTIGUOUS
    // runs, so the inner loop is modulo-free and vectorizes. We vectorize over i (not the k-reduction), so each y[i] still
    // sums in ascending-k order → BIT-IDENTICAL to the scalar cell (the vector mul-then-add is two roundings, not an FMA).
    // <paramref name="x"/> and <paramref name="y"/> MUST be different buffers (we read x while writing y).
    private void AlgApply(double[][] bank, double[] x, double[] y)
    {
        Array.Clear(y, 0, _d);
        var W = Vector<double>.Count;
        for (var k = 0; k < _s; k++)
        {
            var bk = bank[k];
            var n1 = _d - k;                                                    // segment 1: x index = i + k
            var i = 0;
            for (; i <= n1 - W; i += W) (new Vector<double>(bk, i) * new Vector<double>(x, i + k) + new Vector<double>(y, i)).CopyTo(y, i);
            for (; i < n1; i++) y[i] += bk[i] * x[i + k];
            for (i = n1; i <= _d - W; i += W) (new Vector<double>(bk, i) * new Vector<double>(x, i - n1) + new Vector<double>(y, i)).CopyTo(y, i);   // segment 2: the wrap, x index = i - n1
            for (; i < _d; i++) y[i] += bk[i] * x[i - n1];
        }
    }

    /// <summary>Per-thread reusable bump-pool of <c>double[dModel]</c> rows — the backprop hot path allocates ~73 MB of
    /// these per example (activations retained for the backward pass). A training thread processes its shard's examples one
    /// at a time, so it reuses ONE pool across examples: <see cref="Reset"/> at the start of each, then <see cref="Rent"/>
    /// hands out distinct rows (grows on demand). Reset only rewinds the cursor — the arrays live on and are re-handed-out,
    /// so steady-state allocation is ~0. Not thread-safe; one per thread. Rows are dirty unless <see cref="RentZeroed"/>.</summary>
    public sealed class Scratch
    {
        internal readonly int D; internal readonly List<double[]> Rows = new(); internal int Cur;
        internal Scratch(int d) => D = d;
        internal void Reset() => Cur = 0;
        internal double[] Rent() { if (Cur == Rows.Count) Rows.Add(new double[D]); return Rows[Cur++]; }
        internal double[] RentZeroed() { var r = Rent(); Array.Clear(r); return r; }
    }
    /// <summary>A reusable row pool for the training path — create ONE per thread and pass it to <see cref="Accumulate(int[],int,Grads,Scratch)"/>.</summary>
    public Scratch NewScratch() => new(_d);

    private double[][] ApplyAlg(double[][] bank, double[][] x, Scratch? s = null)
    {
        var T = x.Length; var y = new double[T][];
        if (s != null) for (var t = 0; t < T; t++) { y[t] = s.Rent(); AlgApply(bank, x[t], y[t]); }   // pooled: sequential (single thread owns the pool), rows fully written by AlgApply
        else MapPositions(T, t => { y[t] = new double[_d]; AlgApply(bank, x[t], y[t]); });
        return y;
    }

    // single-position algebraic map — one row of ApplyAlg (used by the KV-cache incremental forward).
    private double[] AlgRow(double[][] bank, double[] x) { var y = new double[_d]; AlgApply(bank, x, y); return y; }

    // Σ a[d]·b[d] — SIMD reduction. The horizontal lane-sum reorders the reduction (sub-ULP, deterministic), so this is
    // NOT bit-identical to a scalar dot; used only where the result feeds a softmax/argmax (attention scores, logits).
    private double Dot(double[] a, double[] b)
    {
        var W = Vector<double>.Count; var acc = Vector<double>.Zero; var d = 0;
        for (; d <= _d - W; d += W) acc += new Vector<double>(a, d) * new Vector<double>(b, d);
        var s = Vector.Sum(acc);
        for (; d < _d; d++) s += a[d] * b[d];
        return s;
    }

    // dst[d] += aj·src[d] — SIMD over the OUTPUT index d, so each dst[d] accumulates across calls in the same order as
    // scalar → BIT-IDENTICAL (the attention context sum stays exact).
    private void AddScaled(double[] dst, double[] src, double aj)
    {
        var W = Vector<double>.Count; var va = new Vector<double>(aj); var d = 0;
        for (; d <= _d - W; d += W) (new Vector<double>(src, d) * va + new Vector<double>(dst, d)).CopyTo(dst, d);
        for (; d < _d; d++) dst[d] += aj * src[d];
    }

    /// <summary>Run <paramref name="rowBody"/> for every position 0..T-1 via <see cref="Map"/> — chunked into ~ProcessorCount
    /// disjoint ranges. Sequential unless the model's Map is the EvalApp CPU gate. Each position writes only its own slot.</summary>
    private void MapPositions(int T, Action<int> rowBody)
    {
        var P = T >= 64 ? Math.Min(T, Environment.ProcessorCount) : 1;
        Map.Map(P, 2, c => { var lo = (int)((long)c * T / P); var hi = (int)((long)(c + 1) * T / P); for (var t = lo; t < hi; t++) rowBody(t); });
    }
    // Backward of the algebraic cell, shift-outer + SIMD (modulo-free contiguous runs, same split as AlgApply). Vectorized
    // over the OUTPUT index i: gbank[k][i] += dOut[t][i]·x[t][i+k] and dX[t][i+k] += dOut[t][i]·bank[k][i]. (Gradient
    // accumulation order differs sub-ULP from the old scalar loop — deterministic; gradients aren't a stored format.)
    private void AlgBack(double[][] bank, double[][] gbank, double[][] x, double[][] dOut, double[][] dX)
    {
        var T = x.Length; var W = Vector<double>.Count;
        for (var t = 0; t < T; t++)
        {
            double[] dy = dOut[t], xt = x[t], dxt = dX[t];
            for (var k = 0; k < _s; k++)
            {
                double[] bk = bank[k], gk = gbank[k];
                var n1 = _d - k;
                var i = 0;
                for (; i <= n1 - W; i += W)
                {
                    var vdy = new Vector<double>(dy, i);
                    (new Vector<double>(gk, i) + vdy * new Vector<double>(xt, i + k)).CopyTo(gk, i);
                    (new Vector<double>(dxt, i + k) + vdy * new Vector<double>(bk, i)).CopyTo(dxt, i + k);
                }
                for (; i < n1; i++) { gk[i] += dy[i] * xt[i + k]; dxt[i + k] += dy[i] * bk[i]; }
                for (i = n1; i <= _d - W; i += W)
                {
                    var vdy = new Vector<double>(dy, i);
                    (new Vector<double>(gk, i) + vdy * new Vector<double>(xt, i - n1)).CopyTo(gk, i);
                    (new Vector<double>(dxt, i - n1) + vdy * new Vector<double>(bk, i)).CopyTo(dxt, i - n1);
                }
                for (; i < _d; i++) { gk[i] += dy[i] * xt[i - n1]; dxt[i - n1] += dy[i] * bk[i]; }
            }
        }
    }

    private static double Sig(double x) => 1.0 / (1.0 + Math.Exp(-x));
    private sealed class Cache { public required double[][] X, Q, K, V, Ctx, M, Aval, SigG, Gt, Z; public required double[][] A; }

    private (double[][] h, Cache[] caches) ForwardAll(int[] toks, Scratch? s = null)
    {
        var T = toks.Length;
        double[] Row() => s?.Rent() ?? new double[_d];         // rows fully overwritten before read
        double[] ZRow() => s?.RentZeroed() ?? new double[_d];  // accumulate-target rows (must start at 0)
        var h = new double[T][];
        for (var t = 0; t < T; t++) { h[t] = Row(); double[] e = Emb[toks[t]], pp = Pos[t]; for (var d = 0; d < _d; d++) h[t][d] = e[d] + pp[d]; }
        var caches = new Cache[_layers];
        for (var l = 0; l < _layers; l++)
        {
            var L = Ls[l]; var X = h;
            var q = ApplyAlg(L.Rq, X, s); var k = ApplyAlg(L.Rk, X, s); var v = ApplyAlg(L.Rv, X, s);
            var a = new double[T][]; var ctx = new double[T][]; var inv = 1.0 / Math.Sqrt(_d);
            // CAUSAL: position t attends only to j ≤ t (no lookahead) → generation is KV-cacheable and train/infer
            // consistent. a[t][j] for j > t stays 0, so the backward pass masks itself. Per-position work is independent
            // (each writes only its own a[t]/ctx[t]), so sequential (pooled) == chunked (MapPositions) bit-for-bit.
            void Att(int t)
            {
                a[t] = new double[T];
                var sc = new double[t + 1]; var max = double.NegativeInfinity;
                for (var j = 0; j <= t; j++) { sc[j] = Dot(q[t], k[j]) * inv; if (sc[j] > max) max = sc[j]; }
                var sum = 0.0;
                for (var j = 0; j <= t; j++) { var e = Math.Exp(sc[j] - max); a[t][j] = e; sum += e; }
                for (var j = 0; j <= t; j++) a[t][j] /= sum;
                ctx[t] = ZRow();
                for (var j = 0; j <= t; j++) AddScaled(ctx[t], v[j], a[t][j]);
            }
            if (s != null) for (var t = 0; t < T; t++) Att(t); else MapPositions(T, Att);
            var o = ApplyAlg(L.Ro, ctx, s);
            var m = new double[T][];
            for (var t = 0; t < T; t++) { m[t] = Row(); double[] xt = X[t], ot = o[t]; for (var d = 0; d < _d; d++) m[t][d] = xt[d] + ot[d]; }
            var aval = ApplyAlg(L.A1, m, s); var gt = ApplyAlg(L.Ag, m, s);
            var sig = new double[T][]; var z = new double[T][];
            for (var t = 0; t < T; t++)
            {
                sig[t] = Row(); z[t] = Row();
                if (_bind)
                    for (var c = 0; c + 1 < _d; c += 2) { double ar = aval[t][c], ai = aval[t][c + 1], gr = gt[t][c], gi = gt[t][c + 1]; z[t][c] = ar * gr - ai * gi; z[t][c + 1] = ar * gi + ai * gr; }   // aval ⊛ gt (complex pair multiply = bind)
                else
                    for (var d = 0; d < _d; d++) { sig[t][d] = Sig(gt[t][d]); z[t][d] = aval[t][d] * sig[t][d]; }
            }
            var fz = ApplyAlg(L.Ao, z, s);
            var y = new double[T][];
            for (var t = 0; t < T; t++) { y[t] = Row(); double[] mt = m[t], ft = fz[t]; for (var d = 0; d < _d; d++) y[t][d] = mt[d] + ft[d]; }
            caches[l] = new Cache { X = X, Q = q, K = k, V = v, A = a, Ctx = ctx, M = m, Aval = aval, SigG = sig, Gt = gt, Z = z };
            h = y;
        }
        return (h, caches);
    }

    private double[] Logits(double[] hLast) { var lg = new double[_v]; for (var w = 0; w < _v; w++) lg[w] = C[w] + Dot(Emb[w], hLast); return lg; }

    public int Predict(int[] toks) { var (h, _) = ForwardAll(toks); var lg = Logits(h[^1]); var best = 0; for (var w = 1; w < _v; w++) if (lg[w] > lg[best]) best = w; return best; }

    /// <summary>Next-token logits at the final position (reuses the shared forward path).</summary>
    /// <summary>Next-token logits at the final position (for probability/perplexity evaluation).</summary>
    public double[] LogitsFor(int[] toks) { var (h, _) = ForwardAll(toks); return Logits(h[^1]); }

    /// <summary>Interpretability hook — the "inspectable faces" property. Returns the hidden FACE at the LAST
    /// position as it moves up the residual stream: index 0 is the embedded input to layer 0, index l the input
    /// to layer l, and the final entry the representation the readout actually reads. Every entry is a phasor
    /// face, so it can be decoded (<see cref="PhasorCodec.DecodeSum"/>/<see cref="PhasorCodec.DecodeProduct"/>)
    /// to read what value it carries — letting us watch, layer by layer, where the model lands a computed
    /// answer. Read-only; no effect on training or serving.</summary>
    public double[][] LayerFaces(int[] toks)
    {
        var (h, caches) = ForwardAll(toks);
        var faces = new double[caches.Length + 1][];
        for (var l = 0; l < caches.Length; l++) faces[l] = (double[])caches[l].X[^1].Clone();
        faces[caches.Length] = (double[])h[^1].Clone();
        return faces;
    }

    // ── KV-cache incremental generation ──────────────────────────────────────────────────────────
    /// <summary>Per-generation state: cached per-layer K and V for every position placed so far. Because attention is
    /// CAUSAL, a past position's k/v never changes when a new token is appended, so generation is O(T) per token
    /// (attend the new query against the cache) instead of O(T²) recompute — over a full serve, O(T²) not O(T³).
    /// One cache per generation (not stored on the model), so serving stays thread-safe off the snapshot.</summary>
    public sealed class KvCache
    {
        internal readonly List<double[]>[] K, V;   // [layer] → list of position vectors
        internal int T;                            // positions placed so far (= next position index)
        // reusable per-step scratch — serve is sequential per cache, so a single Step's working vectors can be reused
        // across tokens with NO per-token allocation (lever 2: the model is tiny and stays hot in L1/L2).
        internal readonly double[] H, Q, O, M, Aval, Gt, Z, Ctx, S;
        /// <summary>Positions placed so far — the absolute index the next <see cref="Step"/> will fill.</summary>
        public int Length => T;
        internal KvCache(int layers, int d, int maxT)
        {
            K = new List<double[]>[layers]; V = new List<double[]>[layers];
            for (var l = 0; l < layers; l++) { K[l] = new(); V[l] = new(); }
            H = new double[d]; Q = new double[d]; O = new double[d]; M = new double[d];
            Aval = new double[d]; Gt = new double[d]; Z = new double[d]; Ctx = new double[d]; S = new double[maxT + 1];
        }
        internal void Reset() { for (var l = 0; l < K.Length; l++) { K[l].Clear(); V[l].Clear(); } T = 0; }
    }

    public KvCache NewCache() => new(_layers, _d, _maxT);

    /// <summary>Reset <paramref name="c"/> and run <paramref name="toks"/> through the incremental forward, returning the
    /// next-token logits after the last token. Equivalent to <see cref="LogitsFor"/> (causal) but leaves the cache primed
    /// so subsequent <see cref="Step"/> calls are O(T). <paramref name="toks"/> must be ≤ <see cref="Context"/> long.</summary>
    public double[] Prime(KvCache c, int[] toks)
    {
        c.Reset();
        double[] lg = C;   // fallback for empty input (no context) — should not happen in practice
        foreach (var tok in toks) lg = Step(c, tok);
        return lg;
    }

    /// <summary>Append one token at the next position via the cache and return the next-token logits. O(T) in the current
    /// length. The token's absolute position is <c>c.T</c> (must be &lt; <see cref="Context"/> — the caller slides the window).</summary>
    public double[] Step(KvCache c, int tok)
    {
        var t = c.T;                       // position of this new token
        var pos = Pos[t < _maxT ? t : _maxT - 1];
        double[] h = c.H, q = c.Q, o = c.O, m = c.M, aval = c.Aval, gt = c.Gt, z = c.Z, ctx = c.Ctx, s = c.S;   // all scratch (reused)
        for (var d = 0; d < _d; d++) h[d] = Emb[tok][d] + pos[d];
        var inv = 1.0 / Math.Sqrt(_d);
        for (var l = 0; l < _layers; l++)
        {
            var L = Ls[l];
            AlgApply(L.Rq, h, q);                                             // q from X = h
            var kt = new double[_d]; AlgApply(L.Rk, h, kt);                   // k/v are cached across steps → must be fresh
            var vt = new double[_d]; AlgApply(L.Rv, h, vt);
            var Kl = c.K[l]; var Vl = c.V[l];
            Kl.Add(kt); Vl.Add(vt);        // now length t+1
            var max = double.NegativeInfinity;
            for (var j = 0; j <= t; j++) { s[j] = Dot(q, Kl[j]) * inv; if (s[j] > max) max = s[j]; }
            var sum = 0.0; for (var j = 0; j <= t; j++) { var e = Math.Exp(s[j] - max); s[j] = e; sum += e; }
            Array.Clear(ctx, 0, _d);
            for (var j = 0; j <= t; j++) AddScaled(ctx, Vl[j], s[j] / sum);   // /sum (not *recip) to match ForwardAll bit-for-bit
            AlgApply(L.Ro, ctx, o);
            for (var d = 0; d < _d; d++) m[d] = h[d] + o[d];                  // read h before we reuse it as the output below
            AlgApply(L.A1, m, aval); AlgApply(L.Ag, m, gt);
            if (_bind)
                for (var cc = 0; cc + 1 < _d; cc += 2) { double ar = aval[cc], ai = aval[cc + 1], gr = gt[cc], gi = gt[cc + 1]; z[cc] = ar * gr - ai * gi; z[cc + 1] = ar * gi + ai * gr; }
            else
                for (var d = 0; d < _d; d++) z[d] = aval[d] * Sig(gt[d]);
            AlgApply(L.Ao, z, o);                                            // reuse o as fz (o already consumed into m)
            for (var d = 0; d < _d; d++) h[d] = m[d] + o[d];                 // h ← layer output (h fully read into m above)
        }
        c.T = t + 1;
        return Logits(h);
    }

    /// <summary>Autoregressive generation. Repeatedly predicts the next token over a rolling window of the last
    /// <see cref="Context"/> tokens and appends it. <paramref name="temperature"/> 0 → greedy (argmax); &gt;0 →
    /// sample from softmax(logits/temperature) via a seeded RNG. Returns only the newly generated tokens.</summary>
    public int[] Generate(int[] prompt, int maxNewTokens, double temperature = 0, int seed = 0)
    {
        if (maxNewTokens <= 0) return Array.Empty<int>();
        var rng = temperature > 0 ? new Random(seed) : null;
        var ctx = new List<int>(prompt);
        var outp = new int[maxNewTokens];
        for (var n = 0; n < maxNewTokens; n++)
        {
            var win = ctx.Count > _maxT ? ctx.GetRange(ctx.Count - _maxT, _maxT).ToArray() : ctx.ToArray();
            var lg = LogitsFor(win);
            int next;
            if (rng == null)
            {
                next = 0; for (var w = 1; w < _v; w++) if (lg[w] > lg[next]) next = w;   // greedy argmax
            }
            else
            {
                var max = double.NegativeInfinity; var scaled = new double[_v];
                for (var w = 0; w < _v; w++) { scaled[w] = lg[w] / temperature; if (scaled[w] > max) max = scaled[w]; }
                var sum = 0.0; for (var w = 0; w < _v; w++) { scaled[w] = Math.Exp(scaled[w] - max); sum += scaled[w]; }
                var r = rng.NextDouble() * sum; next = _v - 1;
                for (var w = 0; w < _v; w++) { r -= scaled[w]; if (r <= 0) { next = w; break; } }
            }
            outp[n] = next; ctx.Add(next);
        }
        return outp;
    }

    internal double LossOnly(int[] toks, int answer)
    {
        var (h, _) = ForwardAll(toks); var lg = Logits(h[^1]);
        var max = lg.Max(); var sum = 0.0; for (var w = 0; w < _v; w++) sum += Math.Exp(lg[w] - max);
        return -(lg[answer] - max - Math.Log(sum));
    }

    public double Accumulate(int[] toks, int answer, Grads g) => Accumulate(toks, answer, g, null);

    /// <summary>As <see cref="Accumulate(int[],int,Grads)"/>, but renting the per-position activation/gradient rows from a
    /// reusable per-thread <paramref name="s"/> pool instead of allocating (~73 MB/example otherwise). Bit-identical to the
    /// allocating path (verified by bench --gradcheck): pooled rows are either fully overwritten or explicitly zeroed.</summary>
    public double Accumulate(int[] toks, int answer, Grads g, Scratch? s)
    {
        s?.Reset();
        double[] Row() => s?.Rent() ?? new double[_d];         // fully overwritten before read
        double[] ZRow() => s?.RentZeroed() ?? new double[_d];  // accumulate-target (must start at 0)
        double[] CopyRow(double[] src) { var r = Row(); Array.Copy(src, r, _d); return r; }
        var T = toks.Length;
        var (h, caches) = ForwardAll(toks, s);
        var lg = Logits(h[^1]);
        var max = lg.Max(); var sum = 0.0; var p = new double[_v];
        for (var w = 0; w < _v; w++) { p[w] = Math.Exp(lg[w] - max); sum += p[w]; }
        for (var w = 0; w < _v; w++) p[w] /= sum;
        var loss = -Math.Log(Math.Max(p[answer], 1e-300));

        var dh = new double[T][];
        for (var t = 0; t < T; t++) dh[t] = ZRow();
        for (var w = 0; w < _v; w++)
        {
            var dz = p[w] - (w == answer ? 1.0 : 0.0);
            g.C[w] += dz;
            for (var d = 0; d < _d; d++) { g.Emb[w][d] += dz * h[^1][d]; dh[^1][d] += dz * Emb[w][d]; }
        }
        for (var l = _layers - 1; l >= 0; l--)
        {
            var L = Ls[l]; var G = g.Ls[l]; var cc = caches[l];
            var dM = new double[T][];
            for (var t = 0; t < T; t++) dM[t] = CopyRow(dh[t]);
            var dZ = new double[T][]; for (var t = 0; t < T; t++) dZ[t] = ZRow();
            AlgBack(L.Ao, G.Ao, cc.Z, dh, dZ);
            var dAval = new double[T][]; var dGt = new double[T][];
            for (var t = 0; t < T; t++)
            {
                dAval[t] = Row(); dGt[t] = Row();
                if (_bind)
                    for (var c = 0; c + 1 < _d; c += 2)
                    {
                        double ar = cc.Aval[t][c], ai = cc.Aval[t][c + 1], gr = cc.Gt[t][c], gi = cc.Gt[t][c + 1], zr = dZ[t][c], zi = dZ[t][c + 1];
                        dAval[t][c] = zr * gr + zi * gi; dAval[t][c + 1] = -zr * gi + zi * gr;   // d(a⊛g)/da = conj(g)-style
                        dGt[t][c] = zr * ar + zi * ai; dGt[t][c + 1] = -zr * ai + zi * ar;
                    }
                else
                    for (var d = 0; d < _d; d++) { dAval[t][d] = dZ[t][d] * cc.SigG[t][d]; var ds = dZ[t][d] * cc.Aval[t][d]; dGt[t][d] = ds * cc.SigG[t][d] * (1 - cc.SigG[t][d]); }
            }
            AlgBack(L.A1, G.A1, cc.M, dAval, dM);
            AlgBack(L.Ag, G.Ag, cc.M, dGt, dM);
            var dX = new double[T][];
            for (var t = 0; t < T; t++) dX[t] = CopyRow(dM[t]);
            var dCtx = new double[T][]; for (var t = 0; t < T; t++) dCtx[t] = ZRow();
            AlgBack(L.Ro, G.Ro, cc.Ctx, dM, dCtx);
            var dV = new double[T][]; var dQ = new double[T][]; var dK = new double[T][];
            for (var t = 0; t < T; t++) { dV[t] = ZRow(); dQ[t] = ZRow(); dK[t] = ZRow(); }
            var inv = 1.0 / Math.Sqrt(_d);
            for (var t = 0; t < T; t++)
            {
                // CAUSAL backward: only j ≤ t contributed to ctx[t] (a[t][j]=0 for j>t), so gradients flow to j ≤ t only.
                var dA = new double[t + 1];
                for (var j = 0; j <= t; j++) { var acc = 0.0; for (var d = 0; d < _d; d++) { dV[j][d] += cc.A[t][j] * dCtx[t][d]; acc += dCtx[t][d] * cc.V[j][d]; } dA[j] = acc; }
                var dot = 0.0; for (var j = 0; j <= t; j++) dot += cc.A[t][j] * dA[j];
                for (var j = 0; j <= t; j++) { var dS = cc.A[t][j] * (dA[j] - dot) * inv; if (dS == 0) continue; for (var d = 0; d < _d; d++) { dQ[t][d] += dS * cc.K[j][d]; dK[j][d] += dS * cc.Q[t][d]; } }
            }
            AlgBack(L.Rq, G.Rq, cc.X, dQ, dX);
            AlgBack(L.Rk, G.Rk, cc.X, dK, dX);
            AlgBack(L.Rv, G.Rv, cc.X, dV, dX);
            dh = dX;
        }
        for (var t = 0; t < T; t++) for (var d = 0; d < _d; d++) { g.Emb[toks[t]][d] += dh[t][d]; g.Pos[t][d] += dh[t][d]; }
        return loss;
    }

    public void Step(Grads g, double lr = 1e-3, double scale = 1.0, double b1 = 0.9, double b2 = 0.999, double eps = 1e-8)
    {
        if (_frozen > 0) foreach (var row in g.Emb) Array.Clear(row, 0, _frozen);   // FREEZE identity codec (exact numbers)
        _t++; var c1 = 1 - Math.Pow(b1, _t); var c2 = 1 - Math.Pow(b2, _t);
        foreach (var (param, grad) in Pairs(g))
        {
            if (!_mom.TryGetValue(param, out var mv)) _mom[param] = mv = (new double[param.Length], new double[param.Length]);
            for (var i = 0; i < param.Length; i++) { var gi = grad[i] / scale; if (gi == 0) continue; mv.m[i] = b1 * mv.m[i] + (1 - b1) * gi; mv.v[i] = b2 * mv.v[i] + (1 - b2) * gi * gi; param[i] -= lr * (mv.m[i] / c1) / (Math.Sqrt(mv.v[i] / c2) + eps); }
        }
    }

    public double TrainStep(int[] toks, int answer, double lr = 1e-3) { var g = NewGrads(); var loss = Accumulate(toks, answer, g); Step(g, lr); return loss; }

    /// <summary>Data-parallel epoch (in-box TPL: split batch → per-shard Grads → merge → one Step). Result matches serial.</summary>
    public double TrainEpoch(IReadOnlyList<(int[] Ctx, int Target)> data, int batchSize, double lr, int shuffleSeed, int parallelism = 0)
    {
        var p = parallelism > 0 ? parallelism : Math.Max(1, Environment.ProcessorCount - 1);
        var order = Enumerable.Range(0, data.Count).ToArray();
        var rng = new Random(shuffleSeed);
        for (var i = order.Length - 1; i > 0; i--) { var j = rng.Next(i + 1); (order[i], order[j]) = (order[j], order[i]); }
        double total = 0; var batches = 0;
        for (var start = 0; start < order.Length; start += batchSize)
        {
            var end = Math.Min(start + batchSize, order.Length);
            var shards = new List<(int[], int)>[Math.Min(p, end - start)];
            for (var s = 0; s < shards.Length; s++) shards[s] = new List<(int[], int)>();
            for (var idx = start; idx < end; idx++) shards[(idx - start) % shards.Length].Add(data[order[idx]]);
            var res = new (Grads g, double loss)[shards.Length];
            System.Threading.Tasks.Parallel.For(0, shards.Length, s =>
            {
                var g = NewGrads(); var loss = 0.0;
                foreach (var (c, tgt) in shards[s]) loss += Accumulate(c, tgt, g);
                res[s] = (g, loss);
            });
            var merged = res[0].g; var bl = res[0].loss;
            for (var s = 1; s < res.Length; s++) { merged.Add(res[s].g); bl += res[s].loss; }
            Step(merged, lr, scale: end - start);
            total += bl / (end - start); batches++;
        }
        return batches > 0 ? total / batches : 0;
    }

    /// <summary>Train N epochs with linear LR decay (keeps the multiplicative FFN stable at scale).</summary>
    public void Train(IReadOnlyList<(int[] Ctx, int Target)> data, int epochs, int batchSize = 256, double baseLr = 1e-3, int seed = 1, Action<int, double>? onEpoch = null)
    {
        for (var ep = 1; ep <= epochs; ep++)
        {
            var lr = baseLr * (1.0 - 0.9 * (ep - 1) / Math.Max(1, epochs - 1));
            var loss = TrainEpoch(data, batchSize, lr, seed + ep);
            onEpoch?.Invoke(ep, loss);
        }
    }

    /// <summary>Parameterised finite-difference gradcheck at an arbitrary (larger) config — probes a couple of entries
    /// of EVERY parameter array (so every layer, bank and shift is covered) against central differences.</summary>
    internal static bool GradCheckAt(int vocab, int shifts, int layers, int maxContext, int dModel, int seed, out double worstRel)
    {
        var m = new AlgFormer(vocab, shifts, layers, maxContext, dModel: dModel, frozenPrefix: 0, seed: seed);
        var rng = new Random(seed + 1);
        var toks = Enumerable.Range(0, maxContext).Select(_ => rng.Next(vocab)).ToArray();
        var ans = rng.Next(vocab); const double eps = 1e-5;
        var g = m.NewGrads(); m.Accumulate(toks, ans, g);
        worstRel = 0;
        foreach (var (p, grad) in m.Pairs(g))
            foreach (var i in p.Length > 1 ? new[] { 0, p.Length / 2 } : new[] { 0 })
            {
                var keep = p[i];
                p[i] = keep + eps; var up = m.LossOnly(toks, ans);
                p[i] = keep - eps; var dn = m.LossOnly(toks, ans);
                p[i] = keep;
                var num = (up - dn) / (2 * eps);
                if (Math.Abs(num - grad[i]) > 1e-7) worstRel = Math.Max(worstRel, Math.Abs(num - grad[i]) / (Math.Abs(num) + Math.Abs(grad[i]) + 1e-12));
            }
        return worstRel < 1e-4;
    }

    public static bool GradCheck(out double worstRel) => GradCheckImpl(false, out worstRel);

    /// <summary>Gradcheck the complex-bind FFN path (bindFfn:true) — the phasor bind combine, same probes.</summary>
    public static bool GradCheckBind(out double worstRel) => GradCheckImpl(true, out worstRel);

    private static bool GradCheckImpl(bool bindFfn, out double worstRel)
    {
        var m = new AlgFormer(vocab: 9, shifts: 3, layers: 2, maxContext: 4, dModel: 6, frozenPrefix: 0, seed: 6, bindFfn: bindFfn);
        int[] toks = { 3, 1, 7, 2 }; const int ans = 5; const double eps = 1e-5;
        var g = m.NewGrads(); m.Accumulate(toks, ans, g);
        worstRel = 0;
        var probes = new (double[] p, double[] grad)[]
        {
            (m.Emb[3], g.Emb[3]), (m.Emb[5], g.Emb[5]), (m.Pos[0], g.Pos[0]), (m.C, g.C),
            (m.Ls[0].Rq[1], g.Ls[0].Rq[1]), (m.Ls[1].Rk[0], g.Ls[1].Rk[0]), (m.Ls[0].Rv[2], g.Ls[0].Rv[2]), (m.Ls[1].Ro[1], g.Ls[1].Ro[1]),
            (m.Ls[0].A1[0], g.Ls[0].A1[0]), (m.Ls[0].Ag[2], g.Ls[0].Ag[2]), (m.Ls[1].Ao[1], g.Ls[1].Ao[1]),
        };
        foreach (var (p, grad) in probes)
            for (var i = 0; i < Math.Min(3, p.Length); i++)
            {
                var keep = p[i];
                p[i] = keep + eps; var up = m.LossOnly(toks, ans);
                p[i] = keep - eps; var dn = m.LossOnly(toks, ans);
                p[i] = keep;
                var num = (up - dn) / (2 * eps);
                if (Math.Abs(num - grad[i]) > 1e-7) worstRel = Math.Max(worstRel, Math.Abs(num - grad[i]) / (Math.Abs(num) + Math.Abs(grad[i]) + 1e-12));
            }
        return worstRel < 1e-4;
    }
}
