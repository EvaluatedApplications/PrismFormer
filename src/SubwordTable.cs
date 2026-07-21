// Copyright (c) 2026 Dongyang Stephen Chen, trading as Evaluated Applications. All rights reserved.
// Source-available for study and verification; commercial use, redistribution, or derivatives require permission. See LICENSE.

using System.Reflection;

namespace PrismFormer;

/// <summary>
/// THE committed subword list, loaded once from the embedded resource <c>PrismFormer.subwords.txt</c> (built offline by
/// <see cref="SubwordBuilder"/> / <c>bench --subwords</c>). Embedded so every deployed node carries the byte-identical
/// list — the property that keeps the vocab layout, hence index-based exact-merge, identical across the swarm. If the
/// resource is absent or empty, the list is empty and the model is pure char level (feature dormant).
///
/// <para>Entries are defensively filtered to valid n-grams (len 2..<see cref="SubwordVocab.MaxLen"/>, printable chars),
/// so a malformed resource degrades to fewer subwords rather than crashing the vocab. The filter is deterministic, so
/// all nodes still agree.</para>
/// </summary>
public static class SubwordTable
{
    static readonly string[] _list = Load();
    public static IReadOnlyList<string> List => _list;

    static bool Valid(string w)
    {
        if (w.Length < 2 || w.Length > SubwordVocab.MaxLen) return false;
        foreach (var c in w) if (c < CharVocab.Lo || c > CharVocab.Hi) return false;
        return true;
    }

    static string[] Load()
    {
        try
        {
            using var s = Assembly.GetExecutingAssembly().GetManifestResourceStream("PrismFormer.subwords.txt");
            if (s == null) return Array.Empty<string>();
            using var r = new StreamReader(s);
            var list = new List<string>();
            var seen = new HashSet<string>(StringComparer.Ordinal);
            string? line;
            while ((line = r.ReadLine()) != null)
            {
                // strip only a trailing CR (\r\n files); NEVER trim spaces — leading/trailing space is part of the n-gram
                if (line.EndsWith('\r')) line = line[..^1];
                if (Valid(line) && seen.Add(line)) list.Add(line);
            }
            return list.ToArray();
        }
        catch { return Array.Empty<string>(); }
    }
}
