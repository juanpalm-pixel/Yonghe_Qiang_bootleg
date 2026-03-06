"""
diarize.py
----------
Stage 1: Speaker diarization.

Uses pyannote/speaker-diarization-3.1 to segment the audio into speaker
turns and assigns them to three IntervalTier labels (A, B, C).

The diarization output has empty text labels — it records *who* spoke
*when*, not *what* was said.  Stage 2 fills in the IPA transcription.

Evaluation target: YH-758_1_IPA-SEPERATE-EMPTY.TextGrid
"""

from __future__ import annotations

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torchaudio

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Device selection — GPU if available, otherwise CPU
# ---------------------------------------------------------------------------
def get_device() -> str:
    if torch.cuda.is_available():
        logger.info("CUDA GPU detected — using GPU for diarization.")
        return "cuda"
    logger.info("No CUDA GPU detected — falling back to CPU for diarization.")
    return "cpu"


# ---------------------------------------------------------------------------
# Audio loading helper
# ---------------------------------------------------------------------------
def load_audio_segment(
    audio_path: str,
    start_sec: float = 0.0,
    end_sec: float = 300.0,
    target_sr: int = 16000,
) -> Tuple[torch.Tensor, int]:
    """
    Load a segment of audio from *audio_path* between *start_sec* and *end_sec*.
    Resamples to *target_sr* if necessary.

    Returns (waveform_1d, sample_rate).
    """
    waveform, sr = torchaudio.load(audio_path)

    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
        sr = target_sr

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Trim to requested segment
    start_frame = int(start_sec * sr)
    end_frame = int(end_sec * sr)
    waveform = waveform[:, start_frame:end_frame]

    return waveform, sr


# ---------------------------------------------------------------------------
# Diarization
# ---------------------------------------------------------------------------
def run_diarization(
    audio_path: str,
    hf_token: str,
    start_sec: float = 0.0,
    end_sec: float = 300.0,
    num_speakers: int = 3,
) -> List[Dict]:
    """
    Run pyannote speaker diarization on an audio segment.

    Parameters
    ----------
    audio_path : str
        Path to the WAV file.
    hf_token : str
        HuggingFace token for accessing gated pyannote models.
    start_sec / end_sec : float
        Time boundaries of the segment to analyse.
    num_speakers : int
        Expected number of speakers (forces the diarization to exactly
        this many clusters, helping with consistent A/B/C assignment).

    Returns
    -------
    List of dicts with keys: {"speaker": str, "start": float, "end": float}
    """
    from pyannote.audio import Pipeline

    device = get_device()

    logger.info("Loading pyannote/speaker-diarization-3.1 …")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )
    pipeline = pipeline.to(torch.device(device))

    # Load and save a temporary 300-s segment for pyannote
    waveform, sr = load_audio_segment(audio_path, start_sec, end_sec)
    tmp_dir = Path(audio_path).parent
    tmp_path = tmp_dir / "_tmp_segment.wav"
    torchaudio.save(str(tmp_path), waveform, sr)

    logger.info("Running diarization (this may take several minutes on CPU) …")
    diarization = pipeline(
        str(tmp_path),
        num_speakers=num_speakers,
    )

    # Clean up temporary file
    try:
        tmp_path.unlink()
    except Exception:
        pass

    # Parse diarization result into list of segment dicts
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append(
            {
                "speaker": speaker,
                "start": round(turn.start + start_sec, 6),
                "end": round(turn.end + start_sec, 6),
            }
        )

    segments.sort(key=lambda x: x["start"])
    return segments


# ---------------------------------------------------------------------------
# Map speaker labels → A / B / C
# ---------------------------------------------------------------------------
def map_speakers_to_tiers(
    segments: List[Dict],
    reference_tg=None,
) -> Dict[str, str]:
    """
    Map pyannote speaker cluster labels (e.g. SPEAKER_00) to tier names A/B/C.

    Strategy:
    1. If a reference TextGrid is supplied, match each pyannote cluster to the
       reference tier that has the greatest temporal overlap.
    2. Otherwise, sort speakers by their first appearance time and assign
       A → earliest, B → second, C → third.

    Returns a dict {pyannote_label: tier_name}.
    """
    speaker_labels = sorted(
        {s["speaker"] for s in segments},
        key=lambda sp: min(s["start"] for s in segments if s["speaker"] == sp),
    )

    if reference_tg is not None:
        from textgrid_utils import get_nonempty_intervals

        # Build overlap matrix
        tier_names = reference_tg.tierNames  # ["A", "B", "C"]
        scores = {sp: {t: 0.0 for t in tier_names} for sp in speaker_labels}

        for seg in segments:
            sp = seg["speaker"]
            s_start, s_end = seg["start"], seg["end"]
            for tier_name in tier_names:
                for iv_start, iv_end, iv_text in get_nonempty_intervals(
                    reference_tg, tier_name
                ):
                    overlap = max(0.0, min(s_end, iv_end) - max(s_start, iv_start))
                    scores[sp][tier_name] += overlap

        # Greedy assignment: pick highest overlap, no tier reused
        mapping = {}
        used_tiers = set()
        for sp in speaker_labels:
            best_tier = max(
                (t for t in tier_names if t not in used_tiers),
                key=lambda t: scores[sp][t],
            )
            mapping[sp] = best_tier
            used_tiers.add(best_tier)
    else:
        tier_names_ordered = ["A", "B", "C"]
        mapping = {
            sp: tier_names_ordered[i] for i, sp in enumerate(speaker_labels[:3])
        }

    return mapping


# ---------------------------------------------------------------------------
# Build interval tiers from diarization segments
# ---------------------------------------------------------------------------
def build_empty_textgrid_from_segments(
    segments: List[Dict],
    speaker_map: Dict[str, str],
    duration: float,
) -> Dict[str, List[Tuple[float, float, str]]]:
    """
    Convert a flat list of diarization segments into per-tier interval lists
    with empty text labels.

    Fills gaps between segments with silent (empty) intervals.

    Returns {tier_name: [(start, end, ""), ...]} covering 0..duration.
    """
    tier_names = sorted(set(speaker_map.values()))

    # Group segments by tier
    tier_segments: Dict[str, List[Tuple[float, float]]] = {t: [] for t in tier_names}
    for seg in segments:
        sp = seg["speaker"]
        if sp not in speaker_map:
            continue
        tier = speaker_map[sp]
        tier_segments[tier].append((seg["start"], seg["end"]))

    # Build full interval lists (speech + silence) for each tier
    tier_intervals: Dict[str, List[Tuple[float, float, str]]] = {}
    for tier_name in tier_names:
        intervals = []
        raw = sorted(tier_segments[tier_name])
        cursor = 0.0
        for start, end in raw:
            if start > cursor + 1e-6:
                intervals.append((cursor, start, ""))  # silence gap
            intervals.append((start, end, ""))          # speech region (empty text)
            cursor = end
        if cursor < duration - 1e-6:
            intervals.append((cursor, duration, ""))
        tier_intervals[tier_name] = intervals

    return tier_intervals


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def stage1_diarize(
    audio_path: str,
    hf_token: str,
    duration: float = 300.0,
    reference_tg=None,
) -> Dict[str, List[Tuple[float, float, str]]]:
    """
    Full Stage 1 pipeline:
    1. Run pyannote diarization
    2. Map speaker clusters → A / B / C
    3. Build empty interval tier data

    Returns {tier_name: [(start, end, ""), ...]}
    """
    logger.info("=== Stage 1: Speaker Diarization ===")
    segments = run_diarization(audio_path, hf_token, end_sec=duration)
    speaker_map = map_speakers_to_tiers(segments, reference_tg)
    logger.info("Speaker mapping: %s", speaker_map)
    tier_data = build_empty_textgrid_from_segments(segments, speaker_map, duration)
    return tier_data, segments, speaker_map
