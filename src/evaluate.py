"""
evaluate.py
-----------
Evaluation module: CER (Character Error Rate) and diarization metrics.

Functions
---------
compute_cer(hypothesis, reference)
    Character-level CER using jiwer.

compare_textgrids(hyp_tg, ref_tg)
    Compare two TextGrid objects tier-by-tier, returning per-tier and
    overall CER values.

compute_der(hyp_segments, ref_tg, duration)
    Compute a simple Diarization Error Rate (DER) using segment overlap.

save_cer_report(report, output_path)
    Persist evaluation results to JSON.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Add src to path when imported directly
# ---------------------------------------------------------------------------
_src_dir = Path(__file__).parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))


# ---------------------------------------------------------------------------
# CER helpers
# ---------------------------------------------------------------------------
def compute_cer(hypothesis: str, reference: str) -> float:
    """
    Compute character error rate between *hypothesis* and *reference*.

    CER = (substitutions + insertions + deletions) / len(reference_chars)

    Returns 1.0 when both strings are empty (perfect match), and 1.0 when
    reference is empty but hypothesis is not.
    """
    from jiwer import cer as jiwer_cer

    if not reference.strip() and not hypothesis.strip():
        return 0.0
    if not reference.strip():
        return 1.0

    # jiwer requires non-empty strings
    return float(jiwer_cer(reference, hypothesis))


def compare_textgrids(
    hyp_tg,
    ref_tg,
    strip_tones: bool = False,
) -> Dict:
    """
    Compare *hyp_tg* and *ref_tg* tier-by-tier.

    Parameters
    ----------
    hyp_tg, ref_tg : praatio Textgrid objects
    strip_tones : bool
        If True, strip 2-digit tone numbers before comparison (useful for
        evaluating Stage 2 output against the toneless reference).

    Returns
    -------
    dict with keys:
        "per_tier": {tier_name: {"cer": float, "ref_chars": int, "hyp_chars": int}}
        "overall_cer": float
        "ref_total_chars": int
        "hyp_total_chars": int
    """
    from textgrid_utils import get_intervals, strip_tone_numbers
    import re

    results = {}
    total_ref_chars = 0
    total_hyp_chars = 0
    weighted_cer_sum = 0.0

    for tier_name in ref_tg.tierNames:
        ref_ivs = get_intervals(ref_tg, tier_name)
        hyp_ivs = get_intervals(hyp_tg, tier_name) if tier_name in hyp_tg.tierNames else []

        # Build full text strings from all non-empty intervals (joined with space)
        ref_texts = [t for _, _, t in ref_ivs if t.strip()]
        hyp_texts = [t for _, _, t in hyp_ivs if t.strip()]

        if strip_tones:
            ref_texts = [strip_tone_numbers(t) for t in ref_texts]
            hyp_texts = [strip_tone_numbers(t) for t in hyp_texts]

        ref_str = " ".join(ref_texts)
        hyp_str = " ".join(hyp_texts)

        # Remove spaces for character-level comparison (IPA is written without word spaces)
        ref_chars = ref_str.replace(" ", "")
        hyp_chars = hyp_str.replace(" ", "")

        cer = compute_cer(hyp_chars, ref_chars)
        ref_n = len(ref_chars)
        hyp_n = len(hyp_chars)

        results[tier_name] = {
            "cer": round(cer, 4),
            "ref_chars": ref_n,
            "hyp_chars": hyp_n,
        }

        total_ref_chars += ref_n
        total_hyp_chars += hyp_n
        weighted_cer_sum += cer * ref_n

    overall_cer = weighted_cer_sum / total_ref_chars if total_ref_chars > 0 else 0.0

    return {
        "per_tier": results,
        "overall_cer": round(overall_cer, 4),
        "ref_total_chars": total_ref_chars,
        "hyp_total_chars": total_hyp_chars,
    }


# ---------------------------------------------------------------------------
# Diarization evaluation (Stage 1)
# ---------------------------------------------------------------------------
def compute_segment_stats(
    hyp_segments: List[Dict],
    ref_tg,
    duration: float,
) -> Dict:
    """
    Compute simple diarization evaluation metrics.

    Metrics
    -------
    - boundary_mae_s : mean absolute error (seconds) between hypothesis
      segment boundaries and the nearest reference boundary
    - speech_coverage_pct : percentage of reference speech frames covered
      by hypothesis speech frames (speaker-agnostic)
    - n_hyp_segments : number of diarized segments
    - n_ref_segments : total number of non-empty reference intervals

    Note: A full DER requires matching speaker labels to reference tiers
    (which is done in diarize.py). This function provides an approximation
    suitable for logging purposes.
    """
    from textgrid_utils import get_nonempty_intervals

    # Collect all reference speech intervals (across all tiers)
    ref_speech = []
    for tier_name in ref_tg.tierNames:
        for s, e, _ in get_nonempty_intervals(ref_tg, tier_name):
            ref_speech.append((s, e))
    ref_speech.sort()

    # Collect hypothesis speech intervals
    hyp_speech = [(seg["start"], seg["end"]) for seg in hyp_segments]
    hyp_speech.sort()

    # Speech coverage
    resolution = 0.01  # 10 ms grid
    n_frames = int(duration / resolution)
    ref_mask = np.zeros(n_frames, dtype=bool)
    hyp_mask = np.zeros(n_frames, dtype=bool)

    import numpy as np

    for s, e in ref_speech:
        ref_mask[int(s / resolution) : int(e / resolution)] = True
    for s, e in hyp_speech:
        hyp_mask[int(s / resolution) : int(e / resolution)] = True

    covered = np.logical_and(ref_mask, hyp_mask).sum()
    ref_frames = ref_mask.sum()
    coverage = float(covered / ref_frames) if ref_frames > 0 else 0.0

    # Boundary MAE
    ref_boundaries = sorted({s for s, _ in ref_speech} | {e for _, e in ref_speech})
    hyp_boundaries = sorted({s for s, _ in hyp_speech} | {e for _, e in hyp_speech})

    if ref_boundaries and hyp_boundaries:
        ref_arr = np.array(ref_boundaries)
        hyp_arr = np.array(hyp_boundaries)
        # For each hyp boundary, find nearest ref boundary
        diffs = []
        for hb in hyp_arr:
            nearest = ref_arr[np.argmin(np.abs(ref_arr - hb))]
            diffs.append(abs(float(hb) - float(nearest)))
        mae = float(np.mean(diffs))
    else:
        mae = float("nan")

    return {
        "boundary_mae_s": round(mae, 4),
        "speech_coverage_pct": round(coverage * 100, 2),
        "n_hyp_segments": len(hyp_speech),
        "n_ref_segments": len(ref_speech),
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_cer_report(report: Dict, output_path: str) -> None:
    """Save *report* dict to JSON at *output_path*."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    logger.info("CER report saved to %s", output_path)


def format_cer_summary(report: Dict) -> str:
    """
    Format a short summary string from a CER report dict.
    Suitable for git commit messages and CHANGELOG entries.
    """
    s1 = report.get("stage1", {})
    s2 = report.get("stage2", {})
    s3 = report.get("stage3", {})

    parts = []
    if s1:
        parts.append(
            f"Stage1 coverage={s1.get('speech_coverage_pct', '?')}% "
            f"boundary_mae={s1.get('boundary_mae_s', '?')}s"
        )
    if s2:
        parts.append(f"Stage2 CER={s2.get('overall_cer', '?'):.2%}")
    if s3:
        parts.append(f"Stage3 CER={s3.get('overall_cer', '?'):.2%}")

    return " | ".join(parts)
