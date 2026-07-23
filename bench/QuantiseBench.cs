// Copyright (c) 2026 Dongyang Stephen Chen, trading as Evaluated Applications. All rights reserved.
// Source-available for study and verification; commercial use, redistribution, or derivatives require permission. See LICENSE.

using PrismFormer;

namespace PrismFormer.Bench;

/// <summary>
/// --quantise : DOES THE LEARNED CONTROL QUANTISE INTO CRISP REGIONS? The hypothesis: because the codec substrate keeps
/// every term crisp and quasi-orthogonal, the learned attenuation (softmax attention + sigmoid FFN gates) is FREE to
/// saturate toward hard, near-discrete decisions, whereas a mushy (overlapping) substrate FORCES soft/grey control.
///
/// Test: train a small model on a key→value RECALL task (which rewards crisp routing — attend to the matching key, gate
/// its value through) with two substrates — the CRISP codec vs a deliberately MUSHY one (tokens squished into a low-rank
/// subspace so they heavily overlap) — and probe the gate + attention distributions FRESH vs TRAINED. Prediction: on the
/// crisp substrate the gates pile near 0/1 and attention peaks near 1.0 after training (quantised); the mushy one stays greyer.
/// </summary>
public static class QuantiseBench
{
    public static void Run(int passes = 200)
    {
        try { Console.OutputEncoding = System.Text.Encoding.UTF8; } catch { }
        const int K = 10, pairs = 4, ctx = 16, dm = 128;

        var words = new List<string>(); var idm = new Dictionary<string, int>(StringComparer.Ordinal);
        int Id(string w) { if (idm.TryGetValue(w, out var i)) return i; i = idm.Count; idm[w] = i; words.Add(w); return i; }
        var keys = new int[K]; for (var i = 0; i < K; i++) keys[i] = Id($"k{i}");
        var vals = new int[K]; for (var i = 0; i < K; i++) vals[i] = Id($"v{i}");
        var Q = Id("?");
        var vocab = idm.Count + 4;

        // CRISP: the codec faces (quasi-orthogonal). MUSHY: every token squished into a rank-R random subspace → heavy overlap.
        double[] CrispSeed(int w) { var f = w < words.Count ? PhasorCodec.Encode(words[w]) : new double[PhasorCodec.Dim]; if (dm >= f.Length) return f; var s = new double[dm]; Array.Copy(f, s, dm); return s; }
        const int R = 24; var mr = new Random(99); var basis = new double[R][];
        for (var b = 0; b < R; b++) { basis[b] = new double[dm]; for (var d = 0; d < dm; d++) basis[b][d] = mr.NextDouble() * 2 - 1; }
        double[] MushySeed(int w) { var s = new double[dm]; var wr = new Random(w * 131 + 7); for (var b = 0; b < R; b++) { var c = wr.NextDouble() * 2 - 1; for (var d = 0; d < dm; d++) s[d] += c * basis[b][d]; } return s; }

        AlgFormer Build(Func<int, double[]> seed) => new(vocab, shifts: 8, layers: 3, maxContext: ctx, dModel: dm, frozenPrefix: dm, embedSeed: seed, seed: 1);   // frozenPrefix=dm → substrate stays as-seeded (crisp vs mushy is the only variable)

        (int[] ctx, int tgt) Gen(Random r)
        {
            var ks = Enumerable.Range(0, K).OrderBy(_ => r.Next()).Take(pairs).ToArray();
            var vs = Enumerable.Range(0, K).OrderBy(_ => r.Next()).Take(pairs).ToArray();
            var seq = new List<int>(2 * pairs + 2);
            for (var i = 0; i < pairs; i++) { seq.Add(keys[ks[i]]); seq.Add(vals[vs[i]]); }
            var q = r.Next(pairs); seq.Add(Q); seq.Add(keys[ks[q]]);
            return (seq.ToArray(), vals[vs[q]]);
        }

        var rng = new Random(1);
        var train = Enumerable.Range(0, 800).Select(_ => Gen(rng)).ToList();
        var probes = Enumerable.Range(0, 200).Select(_ => Gen(rng)).ToList();

        (double sat, double grey, double amean, double apeak) Probe(AlgFormer m)
        {
            var gates = new List<double>(); var peaks = new List<double>();
            foreach (var (c, _) in probes) { var (g, a) = m.ProbeControl(c); gates.AddRange(g); peaks.AddRange(a); }
            var sat = gates.Count(x => x < 0.1 || x > 0.9) / (double)Math.Max(1, gates.Count);
            var grey = gates.Count(x => x > 0.4 && x < 0.6) / (double)Math.Max(1, gates.Count);
            var amean = peaks.Count > 0 ? peaks.Average() : 0;
            var apeak = peaks.Count(x => x > 0.9) / (double)Math.Max(1, peaks.Count);
            return (sat, grey, amean, apeak);
        }
        double Acc(AlgFormer m) { var ok = 0; foreach (var (c, t) in probes) if (m.Predict(c) == t) ok++; return ok / (double)probes.Count; }

        Console.WriteLine($"DOES THE CONTROL QUANTISE?  key→value recall, d{dm} L3 S8, {passes} passes  ·  gates near 0/1 & attention peak ≈1 = crisp/quantised\n");
        Console.WriteLine($"  {"substrate",-16}{"recall",8}{"gates 0/1",11}{"gates ~.5",11}{"attn peak",11}{"attn>0.9",10}   state");
        foreach (var (name, seed) in new[] { ("CRISP codec", (Func<int, double[]>)CrispSeed), ("MUSHY rank24", MushySeed) })
        {
            var m = Build(seed);
            var f = Probe(m);
            Console.WriteLine($"  {name,-16}{"(fresh)",8}{f.sat,11:P0}{f.grey,11:P0}{f.amean,11:F2}{f.apeak,10:P0}   untrained");
            for (var ep = 1; ep <= passes; ep++) { var lr = 2e-3 * (1.0 - 0.9 * (ep - 1) / Math.Max(1, passes - 1)); var order = train.OrderBy(_ => rng.Next()); foreach (var (c, t) in order) m.TrainStep(c, t, lr); }
            var tr = Probe(m); var acc = Acc(m);
            Console.WriteLine($"  {name,-16}{acc,8:P0}{tr.sat,11:P0}{tr.grey,11:P0}{tr.amean,11:F2}{tr.apeak,10:P0}   trained\n");
        }
        Console.WriteLine("read: if CRISP-trained shows MORE gates at 0/1 and HIGHER attention peaks than MUSHY-trained (at similar recall),");
        Console.WriteLine("the crisp substrate let the control quantise. If both stay grey / spread, the attenuation is interpolating, not snapping.");
    }
}
