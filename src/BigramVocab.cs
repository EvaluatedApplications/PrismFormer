// Copyright (c) 2026 Dongyang Stephen Chen, trading as Evaluated Applications. All rights reserved.
// Source-available for study and verification; commercial use, redistribution, or derivatives require permission. See LICENSE.

using System.Text;

namespace PrismFormer;

/// <summary>
/// The character vocabulary EXTENDED with a fixed, deterministic list of two-char subword tokens ("bigrams").
/// Ids <c>0..CharVocab.N-1</c> are exactly <see cref="CharVocab"/>'s char ids, so a bigram vocab is a strict
/// SUPERSET of the char vocab: an existing char checkpoint's embedding rows carry over unchanged, and only the
/// appended bigram rows (ids <c>CharVocab.N..</c>) need seeding. This is what lets vocab grow in place (append-only)
/// instead of being a hard fork.
///
/// <para>Because the bigram list is deterministic and identical on every node, the vocab layout is identical
/// everywhere, so the swarm's index-based exact-merge is unaffected: the same token sits at the same embedding row
/// on every node. Growing the list bumps the spec Signature (like growing Shifts/Context), so the whole mesh moves
/// to the new build together.</para>
///
/// <para>Tokenisation is greedy longest-match: at each position, if the next two (folded) chars form a known bigram
/// it emits that bigram token and advances two, else it emits the single char and advances one. Decoding and codec
/// seeding both go through <see cref="Symbol"/>, which returns the literal token text, so a digit bigram like
/// <c>"47"</c> reaches <see cref="PhasorCodec.Encode(string)"/> as a number literal and gets the NUMBER face for free.</para>
///
/// <para>With an EMPTY bigram list this is byte-for-byte the char-level tokenizer (<see cref="Encode"/> == CharVocab
/// char ids, same length), so the feature is fully dormant until a list is supplied.</para>
/// </summary>
public sealed class BigramVocab
{
    readonly string[] _bi;                       // bigrams, fixed order; token id = CharVocab.N + index
    readonly Dictionary<int, int> _biId;         // packed (c1<<8|c2) -> token id, so lookup allocates nothing

    /// <summary>Char ids 0..CharN-1 are the plain char vocab; bigram ids start at CharN.</summary>
    public int CharN => CharVocab.N;             // 96
    public int Size { get; }                     // CharN + bigram count
    public IReadOnlyList<string> Bigrams => _bi;

    /// <summary>Fold any char to its canonical printable form, matching <see cref="CharVocab.Id"/> (printable stays, else space).</summary>
    static char Fold(char c) => c >= CharVocab.Lo && c <= CharVocab.Hi ? c : ' ';
    static int Key(char a, char b) => (a << 8) | b;

    public BigramVocab(IReadOnlyList<string> bigrams)
    {
        _bi = bigrams.ToArray();
        _biId = new Dictionary<int, int>(_bi.Length);
        for (var i = 0; i < _bi.Length; i++)
        {
            var b = _bi[i];
            if (b.Length != 2) throw new ArgumentException($"bigram '{b}' is not exactly two characters");
            if (Fold(b[0]) != b[0] || Fold(b[1]) != b[1]) throw new ArgumentException($"bigram '{b}' contains a non-printable char (must be folded first)");
            if (!_biId.TryAdd(Key(b[0], b[1]), CharVocab.N + i)) throw new ArgumentException($"duplicate bigram '{b}' at index {i}");
        }
        Size = CharVocab.N + _bi.Length;
    }

    /// <summary>Greedy longest-match tokenisation. Identical to CharVocab char-encoding when the bigram list is empty.</summary>
    public int[] Encode(string s)
    {
        var outp = new List<int>(s.Length);
        var i = 0;
        while (i < s.Length)
        {
            var a = Fold(s[i]);
            if (i + 1 < s.Length && _biId.TryGetValue(Key(a, Fold(s[i + 1])), out var bid)) { outp.Add(bid); i += 2; continue; }
            outp.Add(a - CharVocab.Lo);   // single char → id 0..94 (space for anything non-printable)
            i++;
        }
        return outp.ToArray();
    }

    /// <summary>Token id → its literal text (one char, or a two-char bigram). Used for decoding AND for codec seeding.</summary>
    public string Symbol(int id)
    {
        if (id < CharVocab.N) return id == CharVocab.End ? "\n" : ((char)(CharVocab.Lo + Math.Clamp(id, 0, CharVocab.Printable - 1))).ToString();
        var k = id - CharVocab.N;
        return k >= 0 && k < _bi.Length ? _bi[k] : " ";
    }

    public string Decode(IEnumerable<int> ids) { var sb = new StringBuilder(); foreach (var id in ids) sb.Append(Symbol(id)); return sb.ToString(); }
}
