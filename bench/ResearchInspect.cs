// Copyright (c) 2026 Dongyang Stephen Chen, trading as Evaluated Applications. All rights reserved.
// Source-available for study and verification; commercial use, redistribution, or derivatives require permission. See LICENSE.

using PrismFormer;

namespace PrismFormer.Bench;

/// <summary>
/// A targeted, isolated MECHANISTIC study (run: <c>dotnet run --project bench -c Release -- --inspect</c>).
///
/// For each of the two operations that have a phasor-codec homomorphism — addition (linear-phase band) and
/// multiplication (log-phase band) — restricted to the single-digit PRIMITIVE the curriculum teaches directly,
/// we train a PrismFormer and a parameter-matched dense transformer, average over several seeds, and then
/// INSPECT the model. Because activations are phasor faces we can, with no trained probe, (i) decode the
/// last-position face at every depth and watch the answer emerge up the layers, and (ii) decompose WHERE in the
/// final face the answer lives: the homomorphic band alone, the full frozen identity (linear+log), or only after
/// the learned "orbital" tail is included (the model's own readout). That decomposition — impossible on a dense
/// transformer, whose hidden yields nothing without a fitted probe — is the concrete pay-off of inspectable faces.
/// </summary>
public static class ResearchInspect
{
    // vocab: 0..9 integer tokens (id == value); 10 = "+", 11 = "*", 12 = "="
    const int Plus = 10, Mul = 11, Eq = 12, V = 13, MaxV = 9;

    static double[] Seed(int w) =>
        w <= MaxV ? PhasorCodec.NumberFace(w)
        : w == Plus ? PhasorCodec.Encode("+")
        : w == Mul ? PhasorCodec.Encode("*")
        : w == Eq ? PhasorCodec.Encode("=")
        : new double[PhasorCodec.Dim];

    sealed record Op(string Name, int Sym, string Band, Func<int, int, int> Ans, Func<double[], int, int> BandDecode);

    static int[] Seq(Op op, int a, int b) => new[] { a, op.Sym, b, Eq };   // "a <op> b ="  ->  predict Ans(a,b)

    public static void Run(int seeds = 8, int steps = 6000)
    {
        Console.WriteLine("PrismFormer mechanistic probe — isolated single-digit ops, inspecting the faces\n");
        var ops = new[]
        {
            new Op("add", Plus, "linear-phase band", (a, b) => a + b, (f, m) => PhasorCodec.DecodeSum(f, m)),
            new Op("mul", Mul, "log-phase band",    (a, b) => a * b, (f, m) => PhasorCodec.DecodeProduct(f, m)),
        };
        foreach (var op in ops) RunOp(op, seeds, steps);
        Console.WriteLine("reading: 'band' = decode the homomorphic band only; 'identity' = both frozen bands; 'readout' = the");
        Console.WriteLine("model's own prediction (full face, incl. learned orbital). band < identity < readout localises how");
        Console.WriteLine("much of the answer is clean codec algebra vs carried by the learned tail. The transformer number is a");
        Console.WriteLine("linear probe FITTED to its hidden — the only way to read a dense model, and it decodes nothing on its own.");
    }

    static void RunOp(Op op, int seeds, int steps)
    {
        // single-digit-result pairs the curriculum teaches directly (mul needs non-zero operands: log band is undefined at 0)
        var all = new List<(int a, int b)>();
        for (var a = 0; a <= 9; a++)
            for (var b = 0; b <= 9; b++)
            {
                var ans = op.Ans(a, b);
                var ok = ans >= (op.Name == "mul" ? 1 : 0) && ans <= MaxV && (op.Name != "mul" || (a >= 1 && b >= 1));
                if (ok) all.Add((a, b));
            }
        var splitRng = new Random(12345);
        var shuffled = all.OrderBy(_ => splitRng.Next()).ToList();
        var nHeld = Math.Max(4, shuffled.Count / 5);
        var held = shuffled.Take(nHeld).ToList();
        var train = shuffled.Skip(nHeld).ToList();

        var acc = new List<double>(); var xfAcc = new List<double>(); var ablU = new List<double>();
        var bandDec = new List<double>(); var idDec = new List<double>(); var probe = new List<double>();
        double[]? layer = null; long algParams = 0, xfParams = 0;

        for (var s = 0; s < seeds; s++)
        {
            // PrismFormer: dModel = phasor dim so faces are decodable; lean relation-banks
            var alg = new AlgFormer(vocab: V, shifts: 32, layers: 3, maxContext: 8,
                                    dModel: PhasorCodec.Dim, frozenPrefix: PhasorCodec.FrozenReals,
                                    embedSeed: Seed, seed: 100 + s);
            algParams = alg.ParamCount;
            var rA = new Random(200 + s);
            for (var i = 0; i < steps; i++) { var (a, b) = train[rA.Next(train.Count)]; alg.TrainStep(Seq(op, a, b), op.Ans(a, b), 1e-3); }
            acc.Add(held.Count(p => alg.Predict(Seq(op, p.a, p.b)) == op.Ans(p.a, p.b)) / (double)held.Count);

            // ABLATION: same model but frozenPrefix=0 (the numeric identity is LEARNABLE, so the codec
            // homomorphism can drift). Identical data order. Tests whether freezing identity is what generalises.
            var algU = new AlgFormer(vocab: V, shifts: 32, layers: 3, maxContext: 8,
                                     dModel: PhasorCodec.Dim, frozenPrefix: 0,
                                     embedSeed: Seed, seed: 100 + s);
            var rAu = new Random(200 + s);
            for (var i = 0; i < steps; i++) { var (a, b) = train[rAu.Next(train.Count)]; algU.TrainStep(Seq(op, a, b), op.Ans(a, b), 1e-3); }
            ablU.Add(held.Count(p => algU.Predict(Seq(op, p.a, p.b)) == op.Ans(p.a, p.b)) / (double)held.Count);

            // inspect: layer-by-layer band decode + final-face band-vs-identity decomposition (no probe)
            var faces0 = alg.LayerFaces(Seq(op, held[0].a, held[0].b));
            var perLayer = new double[faces0.Length];
            int bandOk = 0, idOk = 0;
            foreach (var (a, b) in held)
            {
                var faces = alg.LayerFaces(Seq(op, a, b));
                var ans = op.Ans(a, b);
                for (var l = 0; l < faces.Length; l++) if (op.BandDecode(faces[l], MaxV) == ans) perLayer[l]++;
                if (op.BandDecode(faces[^1], MaxV) == ans) bandOk++;
                if (DecodeIdentity(faces[^1], MaxV) == ans) idOk++;
            }
            for (var l = 0; l < perLayer.Length; l++) perLayer[l] /= held.Count;
            bandDec.Add(bandOk / (double)held.Count); idDec.Add(idOk / (double)held.Count);
            layer ??= new double[perLayer.Length];
            for (var l = 0; l < perLayer.Length; l++) layer[l] += perLayer[l] / seeds;

            // parameter-matched transformer + a linear probe on its opaque hidden
            var xf = new MiniTransformer(vocab: V, dModel: 72, dff: 144, layers: 4, maxT: 8, seed: 300 + s);
            xfParams = xf.ParamCount;
            var rX = new Random(400 + s);
            for (var i = 0; i < steps; i++) { var (a, b) = train[rX.Next(train.Count)]; xf.TrainStep(Seq(op, a, b), op.Ans(a, b), 1e-3); }
            xfAcc.Add(held.Count(p => xf.Predict(Seq(op, p.a, p.b)) == op.Ans(p.a, p.b)) / (double)held.Count);
            probe.Add(LinearProbe(op, xf, train, held));
        }

        var (am, asd) = MS(acc); var (xm, xsd) = MS(xfAcc); var (um, usd) = MS(ablU);
        var (bm, bsd) = MS(bandDec); var (im, isd) = MS(idDec); var (pm, psd) = MS(probe);

        Console.WriteLine($"=== {op.Name.ToUpper()} (single-digit result) · {train.Count} train / {held.Count} held · {seeds} seeds × {steps} steps · params Prism {algParams:N0} / xf {xfParams:N0} ===");
        Console.WriteLine($"  held-out accuracy:   PrismFormer {am,6:P1} ± {asd:P1}    transformer {xm,6:P1} ± {xsd:P1}");
        Console.WriteLine($"  ablation (frozen identity): frozen {am,6:P1} ± {asd:P1}   vs unfrozen {um,6:P1} ± {usd:P1}   (freezing the numeric identity is what generalises)");
        Console.WriteLine($"  {op.Band} decode, up the layers (no probe):");
        for (var l = 0; l < layer!.Length; l++)
        {
            var lbl = l == 0 ? "input" : l == layer.Length - 1 ? "final" : $"layer {l}";
            Console.WriteLine($"      {lbl,-8} {layer[l],6:P1}");
        }
        Console.WriteLine($"  where the answer lives in the FINAL face:");
        Console.WriteLine($"      band     {bm,6:P1} ± {bsd:P1}   ({op.Band} only, no probe)");
        Console.WriteLine($"      identity {im,6:P1} ± {isd:P1}   (both frozen bands, no probe)");
        Console.WriteLine($"      readout  {am,6:P1} ± {asd:P1}   (full face incl. learned orbital = the model's own prediction)");
        Console.WriteLine($"  transformer, fitted linear probe on hidden: {pm,6:P1} ± {psd:P1}\n");
    }

    // Decode a value from the full FROZEN IDENTITY (linear + log bands = first FrozenReals reals), no probe.
    static int DecodeIdentity(double[] h, int max)
    {
        int best = 0; var bz = double.NegativeInfinity;
        for (var c = 0; c <= max; c++)
        {
            var f = PhasorCodec.NumberFace(c);
            var s = 0.0; for (var i = 0; i < PhasorCodec.FrozenReals; i++) s += h[i] * f[i];
            if (s > bz) { bz = s; best = c; }
        }
        return best;
    }

    // Softmax linear probe over 0..MaxV, trained on the transformer's last-hidden features.
    static double LinearProbe(Op op, MiniTransformer xf, List<(int a, int b)> train, List<(int a, int b)> held)
    {
        var C = MaxV + 1;
        var d = xf.LastHidden(Seq(op, 0, 0)).Length;
        var W = Enumerable.Range(0, C).Select(_ => new double[d]).ToArray();
        var bias = new double[C];
        var X = train.Select(p => xf.LastHidden(Seq(op, p.a, p.b))).ToArray();
        var Y = train.Select(p => op.Ans(p.a, p.b)).ToArray();

        for (var ep = 0; ep < 300; ep++)
            for (var i = 0; i < X.Length; i++)
            {
                var logit = new double[C];
                for (var c = 0; c < C; c++) { var z = bias[c]; var w = W[c]; for (var k = 0; k < d; k++) z += w[k] * X[i][k]; logit[c] = z; }
                var mx = logit.Max(); var sum = 0.0; var pr = new double[C];
                for (var c = 0; c < C; c++) { pr[c] = Math.Exp(logit[c] - mx); sum += pr[c]; }
                for (var c = 0; c < C; c++) pr[c] /= sum;
                for (var c = 0; c < C; c++)
                {
                    var g = pr[c] - (c == Y[i] ? 1.0 : 0.0);
                    bias[c] -= 0.05 * g; var w = W[c];
                    for (var k = 0; k < d; k++) w[k] -= 0.05 * g * X[i][k];
                }
            }

        var ok = 0;
        foreach (var (a, b) in held)
        {
            var h = xf.LastHidden(Seq(op, a, b));
            var best = 0; var bl = double.NegativeInfinity;
            for (var c = 0; c < C; c++) { var z = bias[c]; var w = W[c]; for (var k = 0; k < d; k++) z += w[k] * h[k]; if (z > bl) { bl = z; best = c; } }
            if (best == op.Ans(a, b)) ok++;
        }
        return ok / (double)held.Count;
    }

    static (double mean, double sd) MS(List<double> xs)
    {
        var m = xs.Average();
        var sd = xs.Count > 1 ? Math.Sqrt(xs.Sum(x => (x - m) * (x - m)) / (xs.Count - 1)) : 0.0;
        return (m, sd);
    }
}
