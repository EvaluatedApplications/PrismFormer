// Copyright (c) 2026 Dongyang Stephen Chen, trading as Evaluated Applications. All rights reserved.
// Source-available for study and verification; commercial use, redistribution, or derivatives require permission. See LICENSE.

using System.Text;

namespace PrismFormer;

/// <summary>
/// The character vocabulary EXTENDED with a fixed, deterministic list of variable-length subword tokens (n-grams of
/// 2..4 chars). Ids <c>0..CharVocab.N-1</c> are exactly <see cref="CharVocab"/>'s char ids, so a subword vocab is a
/// strict SUPERSET of the char vocab: an existing char checkpoint's embedding rows carry over unchanged, and only the
/// appended subword rows (ids <c>CharVocab.N..</c>) need seeding. This is what lets vocab grow in place (append-only)
/// instead of being a hard fork.
///
/// <para>Because the subword list is deterministic and identical on every node, the vocab layout is identical
/// everywhere, so the swarm's index-based exact-merge is unaffected: the same token sits at the same embedding row on
/// every node. Growing the list bumps the spec Signature (like growing Shifts/Context), so the whole mesh moves to the
/// new build together.</para>
///
/// <para>Tokenisation is greedy LONGEST-match: at each position it tries the longest n-gram first (down to a single
/// char). Decoding and codec seeding both go through <see cref="Symbol"/>, which returns the literal token text, so a
/// numeric token like <c>"47"</c>, <c>"470"</c> or <c>"4700"</c> reaches <see cref="PhasorCodec.Encode(string)"/> as a
/// number literal and gets the NUMBER face (value-as-phase, arithmetic-capable) for free, up to 4 digits.</para>
///
/// <para>With an EMPTY list this is byte-for-byte the char-level tokenizer (<see cref="Encode"/> == CharVocab char ids,
/// same length), so the feature is fully dormant until a list is supplied.</para>
/// </summary>
public sealed class SubwordVocab
{
    public const int MaxLen = 4;                 // longest n-gram we tokenize (2..4)
    readonly string[] _sw;                       // subwords, fixed order; token id = CharVocab.N + index
    readonly Dictionary<string, int> _id;        // n-gram text -> token id
    readonly int _maxLen;                        // actual longest present (<= MaxLen), to bound the match probe

    /// <summary>Char ids 0..CharN-1 are the plain char vocab; subword ids start at CharN.</summary>
    public int CharN => CharVocab.N;             // 96
    public int Size { get; }                     // CharN + subword count
    public IReadOnlyList<string> Subwords => _sw;

    /// <summary>Fold any char to its canonical printable form, matching <see cref="CharVocab.Id"/> (printable stays, else space).</summary>
    static char Fold(char c) => c >= CharVocab.Lo && c <= CharVocab.Hi ? c : ' ';

    public SubwordVocab(IReadOnlyList<string> subwords)
    {
        _sw = subwords.ToArray();
        _id = new Dictionary<string, int>(_sw.Length, StringComparer.Ordinal);
        _maxLen = 1;
        for (var i = 0; i < _sw.Length; i++)
        {
            var w = _sw[i];
            if (w.Length < 2 || w.Length > MaxLen) throw new ArgumentException($"subword '{w}' length {w.Length} out of range 2..{MaxLen}");
            foreach (var ch in w) if (Fold(ch) != ch) throw new ArgumentException($"subword '{w}' contains a non-printable char (must be folded first)");
            if (!_id.TryAdd(w, CharVocab.N + i)) throw new ArgumentException($"duplicate subword '{w}' at index {i}");
            if (w.Length > _maxLen) _maxLen = w.Length;
        }
        Size = CharVocab.N + _sw.Length;
    }

    /// <summary>Greedy longest-match tokenisation (tries len _maxLen..2, then the single char). Identical to CharVocab
    /// char-encoding when the list is empty.</summary>
    public int[] Encode(string s)
    {
        // fold once so matches are over canonical printable chars (consistent with how the list was counted)
        var f = new char[s.Length];
        for (var i = 0; i < s.Length; i++) f[i] = Fold(s[i]);
        var text = new string(f);

        var outp = new List<int>(s.Length);
        var pos = 0;
        while (pos < text.Length)
        {
            var matched = false;
            var hi = Math.Min(_maxLen, text.Length - pos);
            for (var len = hi; len >= 2; len--)
                if (_id.TryGetValue(text.Substring(pos, len), out var id)) { outp.Add(id); pos += len; matched = true; break; }
            if (!matched) { outp.Add(text[pos] - CharVocab.Lo); pos++; }   // single char → id 0..94
        }
        return outp.ToArray();
    }

    /// <summary>Token id → its literal text (one char, or a 2..4-char subword). Used for decoding AND for codec seeding.</summary>
    public string Symbol(int id)
    {
        if (id < CharVocab.N) return id == CharVocab.End ? "\n" : ((char)(CharVocab.Lo + Math.Clamp(id, 0, CharVocab.Printable - 1))).ToString();
        var k = id - CharVocab.N;
        return k >= 0 && k < _sw.Length ? _sw[k] : " ";
    }

    public string Decode(IEnumerable<int> ids) { var sb = new StringBuilder(); foreach (var id in ids) sb.Append(Symbol(id)); return sb.ToString(); }
}
