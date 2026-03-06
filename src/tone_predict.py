"""
tone_predict.py
---------------
Stage 3: Add tone numbers to IPA transcription.

Uses librosa.pyin (probabilistic YIN) to estimate the fundamental frequency
(F0) contour for each speech interval, then applies a rule-based classifier
to assign one of the four attested Yonghe Qiang tones:

    55  — high level       (high mean F0, near-zero slope)
    53  — high falling     (high mean F0, negative slope)
    35  — low rising       (low mean F0, positive slope)
    33  — mid level        (mid mean F0, near-zero slope)

Tone numbers are appended directly after each IPA syllable in the style
used by the reference TextGrid, e.g.:
    "tɕəw" → "tɕəw35"
    "tə paj ʂaj" → "tə33paj33ʂaj53"

Syllable segmentation
----------------------
A rough syllable boundary is inferred from the IPA string: each onset
consonant cluster plus the following vowel nucleus (and optional coda) is
treated as one syllable.  Tone classification is applied uniformly to all
syllables in the interval (the F0 contour for the whole interval is used).

This is a deliberate simplification for the zero-shot bootstrapping scenario.

Evaluation target: YH-758_3_IPA-SEPERATE-WITH_TONES.TextGrid
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

TARGET_SR = 16_000

# ---------------------------------------------------------------------------
# IPA vowel / consonant inventories (for syllable splitting)
# ---------------------------------------------------------------------------
IPA_VOWELS = "aɑæɛeəɤiɨoɔuyrɹ"  # characters that can be vowel nuclei
IPA_TONES = re.compile(r"\d{2}")   # matches existing 2-digit tone numbers


# ---------------------------------------------------------------------------
# F0 extraction
# ---------------------------------------------------------------------------
def extract_f0(audio_array: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """
    Estimate F0 (Hz) using librosa.pyin.
    Returns an array of F0 values (NaN where unvoiced).
    """
    import librosa

    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio_array,
        fmin=librosa.note_to_hz("C2"),   # ~65 Hz
        fmax=librosa.note_to_hz("C7"),   # ~2093 Hz
        sr=sr,
        frame_length=2048,
        hop_length=256,
    )
    return f0  # shape: (n_frames,), NaN where unvoiced


# ---------------------------------------------------------------------------
# Tone classification
# ---------------------------------------------------------------------------
def classify_tone(f0: np.ndarray) -> str:
    """
    Classify the pitch contour *f0* into one of {33, 35, 53, 55}.

    Uses voiced frames only (non-NaN).  Falls back to '33' when there
    are insufficient voiced frames.

    Strategy
    --------
    We use the *start* F0 (first third of voiced frames) and *end* F0
    (last third) rather than the overall mean, because tones like 53
    (high-falling) start high even though their mean may be mid-range.

    ┌──────────────┬──────────────┬──────┐
    │  start_norm  │ end_start_δ  │ tone │
    ├──────────────┼──────────────┼──────┤
    │    ≥ 0.5     │   ≤ -0.1    │  53  │  high-falling
    │    ≥ 0.5     │   > -0.1    │  55  │  high-level
    │    < 0.5     │   ≥  0.1    │  35  │  low-rising
    │    < 0.5     │   < 0.1     │  33  │  mid-level
    └──────────────┴──────────────┴──────┘

    *start_norm* = start_f0 normalised to [0,1] within the voiced range.
    *end_start_δ* = (end_f0 − start_f0) / voiced_range  (positive = rising).
    """
    voiced = f0[~np.isnan(f0)]
    if len(voiced) < 3:
        return "33"  # default: mid level

    third = max(1, len(voiced) // 3)
    start_hz = float(voiced[:third].mean())
    end_hz = float(voiced[-third:].mean())

    f0_min, f0_max = voiced.min(), voiced.max()
    f0_range = float(f0_max - f0_min) if f0_max > f0_min else 1.0

    # Normalised start height: 0 = lowest, 1 = highest in this segment
    start_norm = (start_hz - f0_min) / f0_range

    # Direction: positive = rising, negative = falling
    delta = (end_hz - start_hz) / f0_range

    # When the F0 barely varies, the intra-segment normalisation cannot
    # distinguish 55 (high-level) from 33 (mid-level).
    # "Flat" = range is less than 20 Hz OR less than 10% of the mean F0.
    mean_hz = float(voiced.mean())
    flat_threshold = max(20.0, 0.10 * mean_hz)
    is_flat = f0_range < flat_threshold

    if is_flat:
        return "55" if mean_hz > 170.0 else "33"

    if start_norm >= 0.5:
        return "53" if delta <= -0.1 else "55"
    else:
        return "35" if delta >= 0.1 else "33"


# ---------------------------------------------------------------------------
# Syllable segmentation of IPA string
# ---------------------------------------------------------------------------
def segment_syllables(ipa: str) -> List[str]:
    """
    Roughly split an IPA string into syllables.

    Rules (applied left-to-right):
    1. A syllable starts with an optional consonant cluster (non-vowel chars).
    2. A vowel nucleus follows (one or more vowel characters).
    3. An optional coda of non-vowel characters before the next vowel or end.

    Falls back to treating the entire string as a single syllable if the
    pattern cannot be detected (e.g. all-consonant or empty strings).
    """
    ipa_clean = IPA_TONES.sub("", ipa).strip()
    if not ipa_clean:
        return []

    syllables = []
    i = 0
    n = len(ipa_clean)

    while i < n:
        # Skip spaces and commas — treat as syllable boundaries
        if ipa_clean[i] in " ,":
            i += 1
            continue

        syl = []

        # Onset: consume non-vowel characters
        while i < n and ipa_clean[i] not in IPA_VOWELS and ipa_clean[i] not in " ,":
            syl.append(ipa_clean[i])
            i += 1

        # Nucleus: consume vowel characters
        nucleus_start = i
        while i < n and ipa_clean[i] in IPA_VOWELS:
            syl.append(ipa_clean[i])
            i += 1

        # Coda: consume non-vowel characters until next vowel or space
        while i < n and ipa_clean[i] not in IPA_VOWELS and ipa_clean[i] not in " ,":
            # Peek ahead — if next char is a vowel this belongs to the next onset
            if i + 1 < n and ipa_clean[i + 1] in IPA_VOWELS:
                break
            syl.append(ipa_clean[i])
            i += 1

        if syl:
            syllables.append("".join(syl))

    # Fallback
    if not syllables:
        syllables = [ipa_clean]

    return syllables


def insert_tones(ipa_text: str, tone: str) -> str:
    """
    Append *tone* after each syllable in *ipa_text*.

    Words (space-separated tokens) are treated as individual tone-carrying
    units. Each word gets the same tone label (whole-interval classification).

    Example
    -------
    >>> insert_tones("tɕəw", "35")
    'tɕəw35'
    >>> insert_tones("tə paj ʂaj", "33")
    'tə33paj33ʂaj53'   # (tone 53 on last syllable is illustrative — actual
                        #  output would be uniform '33' in simple mode)
    """
    if not ipa_text.strip():
        return ipa_text

    syllables = segment_syllables(ipa_text)
    if not syllables:
        return ipa_text

    # Attach the same tone to every syllable
    return "".join(s + tone for s in syllables)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def stage3_add_tones(
    audio_path: str,
    tier_data: Dict[str, List[Tuple[float, float, str]]],
) -> Dict[str, List[Tuple[float, float, str]]]:
    """
    Full Stage 3 pipeline.

    Parameters
    ----------
    audio_path : str
        Path to the WAV file.
    tier_data : dict
        Output from Stage 2: {tier_name: [(start, end, ipa_text), ...]}

    Returns
    -------
    dict
        {tier_name: [(start, end, ipa_with_tones), ...]}
    """
    import torchaudio

    logger.info("=== Stage 3: Tone Prediction ===")

    # Load full waveform once
    waveform, sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        import torchaudio.functional as F
        waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)
        sr = TARGET_SR
    audio_np = waveform[0].numpy()

    result: Dict[str, List[Tuple[float, float, str]]] = {}

    for tier_name, intervals in tier_data.items():
        new_intervals = []
        for start, end, ipa_text in intervals:
            if not ipa_text.strip():
                new_intervals.append((start, end, ipa_text))
                continue

            duration = end - start
            if duration < 0.05:
                new_intervals.append((start, end, ipa_text + "33"))
                continue

            # Extract audio segment
            s_frame = int(start * TARGET_SR)
            e_frame = int(end * TARGET_SR)
            segment = audio_np[s_frame:e_frame]

            # F0 extraction
            try:
                f0 = extract_f0(segment, sr=TARGET_SR)
                tone = classify_tone(f0)
            except Exception as exc:
                logger.warning("F0 extraction failed for [%s] %.2f–%.2f: %s", tier_name, start, end, exc)
                tone = "33"

            toned_text = insert_tones(ipa_text, tone)
            logger.debug("[%s] %.2f–%.2f  tone=%s  %r → %r", tier_name, start, end, tone, ipa_text, toned_text)
            new_intervals.append((start, end, toned_text))

        result[tier_name] = new_intervals

    return result
