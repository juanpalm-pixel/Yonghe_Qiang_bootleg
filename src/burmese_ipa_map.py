"""
burmese_ipa_map.py
------------------
Handcrafted mapping from Burmese Unicode characters / common sequences
to their IPA equivalents, calibrated toward the phoneme inventory of
Yonghe Qiang (yong1287).

Background
----------
Both Burmese and Yonghe Qiang are Tibeto-Burman languages. While they
belong to different sub-branches (Lolo-Burmese vs. Qiangic), their
consonant and vowel inventories share enough overlap that the Burmese
acoustic model provides a reasonable zero-shot starting point.

The mapping below is an *approximation*. Burmese phonemes that have no
direct equivalent in the Yonghe Qiang inventory are mapped to the closest
available IPA symbol. Accuracy improves with supervised fine-tuning.

Usage
-----
    from burmese_ipa_map import burmese_to_ipa

    ipa = burmese_to_ipa("မင်္ဂလာ")
"""

from __future__ import annotations

import re
import unicodedata

# ---------------------------------------------------------------------------
# Multi-character sequences (digraphs / trigraphs) — checked FIRST
# ---------------------------------------------------------------------------
MULTI_CHAR: dict[str, str] = {
    # Aspiration marker sequences
    "\u1062\u102B": "a",           # vowel combinations
    "\u1000\u103B\u102D": "tɕi",
    "\u1001\u103B\u102D": "tɕʰi",
    "\u1000\u1039\u1000": "kk",
    "\u1000\u1039\u1002": "km",
    # Common Burmese syllable finals mapped to IPA codas
    "\u102B\u1037": "aN",          # -an nasal
    "\u102C\u1037": "aN",
    "\u1004\u103A\u1037": "ʔ",     # aung → glottal
    "\u1014\u103A": "N",           # final nasal
    "\u1019\u103A": "m",
    "\u1004\u103A": "N",
    "\u1009\u103A": "N",
    "\u1010\u103A": "t",
    "\u1015\u103A": "p",
    "\u1000\u103A": "k",
    # Medial clusters
    "\u103B\u102D": "ji",
    "\u103C\u102D": "wi",
    "\u103B\u102C": "ja",
    "\u103C\u102C": "wa",
}

# ---------------------------------------------------------------------------
# Single Unicode code-point → IPA
# ---------------------------------------------------------------------------
SINGLE_CHAR: dict[str, str] = {
    # ---- Burmese consonants (initials) ----
    "\u1000": "k",      # က  k
    "\u1001": "kʰ",     # ခ  kh
    "\u1002": "ɡ",      # ဂ  g
    "\u1003": "ŋ",      # ဃ  gh → ŋ
    "\u1004": "ŋ",      # င  ng
    "\u1005": "tɕ",     # စ  c
    "\u1006": "tɕʰ",    # ဆ  ch
    "\u1007": "dʑ",     # ဇ  j
    "\u1008": "z",      # ဈ  jh → z
    "\u1009": "ɲ",      # ဉ  ny
    "\u100A": "ɲ",      # ည  ny
    "\u100B": "t",      # ဋ  ṭ
    "\u100C": "tʰ",     # ဌ  ṭh
    "\u100D": "d",      # ဍ  ḍ
    "\u100E": "dz",     # ဎ  ḍh
    "\u100F": "n",      # ဏ  ṇ
    "\u1010": "t",      # တ  t
    "\u1011": "tʰ",     # ထ  th
    "\u1012": "d",      # ဒ  d
    "\u1013": "dz",     # ဓ  dh
    "\u1014": "n",      # န  n
    "\u1015": "p",      # ပ  p
    "\u1016": "pʰ",     # ဖ  ph
    "\u1017": "b",      # ဗ  b
    "\u1018": "b",      # ဘ  bh
    "\u1019": "m",      # မ  m
    "\u101A": "j",      # ယ  y
    "\u101B": "ɹ",      # ရ  r → ɹ
    "\u101C": "l",      # လ  l
    "\u101D": "w",      # ဝ  w
    "\u101E": "θ",      # သ  th (dental) → θ, approximated
    "\u101F": "h",      # ဟ  h
    "\u1020": "l",      # ဠ  ḷ
    "\u1021": "ʔ",      # အ  glottal stop onset
    # ---- Burmese vowels / diacritics ----
    "\u102A": "aw",     # ာ  aa
    "\u102B": "a",      # ာ  aa (short)
    "\u102C": "a",      # ာ  aa
    "\u102D": "i",      # ိ   i
    "\u102E": "iː",     # ီ   ii
    "\u102F": "u",      # ု   u
    "\u1030": "uː",     # ူ   uu
    "\u1031": "e",      # ေ  e
    "\u1032": "ɛ",      # ဲ   ai → ɛ
    "\u1036": "N",      # ံ   anusvara → nasal
    "\u1037": "ʔ",      # ့   creaky voice marker → ʔ
    "\u1038": "",       # း   visarga (tone marker, not phoneme — drop)
    "\u1039": "",       # ် killed letter (silent asat in stacked consonants — skip)
    "\u103A": "",       # ်   asat (final consonant marker — handled via MULTI_CHAR)
    "\u103B": "j",      # ျ  ya-pin medial
    "\u103C": "w",      # ွ   wa-hswe medial
    "\u103D": "w",      # ှ   ha-hto medial
    "\u103E": "h",      # ှ   ha-hto → aspiration
    "\u1040": "0",      # ၀ Burmese digit — drop
    # Burmese digits 1–9 — drop
    **{chr(0x1041 + i): "" for i in range(9)},
    # Punctuation — drop
    "\u104A": " ",      # ၊  minor pause
    "\u104B": " ",      # ။  full stop
}

# Characters to silently discard
DISCARD: set[str] = {
    "\u200B",   # zero-width space
    "\u200C",   # zero-width non-joiner
    "\u200D",   # zero-width joiner
    "\uFEFF",   # BOM
}


def burmese_to_ipa(text: str) -> str:
    """
    Convert a Burmese Unicode string to an approximate IPA string.

    The conversion applies multi-character substitutions first, then
    single-character substitutions. Unknown characters are dropped with
    a warning comment appended for debugging.
    """
    if not text:
        return ""

    result = []
    i = 0
    n = len(text)

    while i < n:
        # Try longest multi-char match first (up to 3 chars)
        matched = False
        for length in (3, 2):
            if i + length <= n:
                chunk = text[i : i + length]
                if chunk in MULTI_CHAR:
                    result.append(MULTI_CHAR[chunk])
                    i += length
                    matched = True
                    break

        if not matched:
            ch = text[i]
            if ch in DISCARD:
                i += 1
                continue
            ipa = SINGLE_CHAR.get(ch)
            if ipa is not None:
                result.append(ipa)
            else:
                # Unknown character — keep as-is with a marker for review
                cat = unicodedata.category(ch)
                if cat.startswith("L"):  # Letter
                    result.append(f"[?{ch}]")
            i += 1

    # Post-process: collapse whitespace, remove empty brackets from digit drops
    out = "".join(result)
    out = re.sub(r"\[?\??\]?", "", out)   # remove leftover [] artifacts
    out = re.sub(r"\s+", " ", out).strip()
    return out


# ---------------------------------------------------------------------------
# Yonghe Qiang phoneme inventory (for reference / validation)
# ---------------------------------------------------------------------------
YQ_CONSONANTS = {
    "p", "pʰ", "b", "m", "t", "tʰ", "d", "n", "ts", "tsʰ", "dz", "s", "z",
    "tʂ", "tʂʰ", "dʐ", "ʂ", "ʐ", "tɕ", "tɕʰ", "dʑ", "ɕ", "ʑ", "k", "kʰ",
    "g", "ŋ", "l", "ɹ", "j", "w", "ɦ", "h", "ʔ", "N",
}

YQ_VOWELS = {
    "a", "ɑ", "æ", "ɛ", "e", "ə", "ɤ", "i", "ɨ", "o", "ɔ", "u", "y", "ɹ",
}

YQ_TONES = {"33", "35", "53", "55"}
