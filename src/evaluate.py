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

import numpy as np

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
# SER and StER (Segment Error Rate / Segment Time Error Rate)
# ---------------------------------------------------------------------------
def compute_ser_ster(
    hyp_segs_per_tier: Dict[str, List[Tuple[float, float]]],
    ref_tg,
    iou_threshold: float = 0.3,
) -> Dict:
    """
    Compute Segment Error Rate (SER) and Segment Time Error Rate (StER).

    SER  = Σ(added_segments + deleted_segments) / Total_ref_segments
    StER = Σ(added_secs + deleted_secs per segment) / Total_ref_segments

    Matching: greedily pair hyp↔ref segments by IoU ≥ *iou_threshold*.
    Unmatched hyp segments count as added; unmatched ref segments as deleted.

    Parameters
    ----------
    hyp_segs_per_tier : dict
        {tier_name: [(start, end), ...]} — speech intervals only from Stage 1.
    ref_tg : praatio Textgrid
        Reference TextGrid (use Stage 2 reference so non-empty text marks speech).
    iou_threshold : float
        IoU required to consider a hyp and ref segment as matching.
    """
    from textgrid_utils import get_nonempty_intervals

    total_ref = 0
    total_added = 0
    total_deleted = 0
    total_added_secs = 0.0
    total_deleted_secs = 0.0
    per_tier: Dict[str, Dict] = {}

    for tier_name in ref_tg.tierNames:
        ref_ivs = [(s, e) for s, e, _ in get_nonempty_intervals(ref_tg, tier_name)]
        hyp_ivs = hyp_segs_per_tier.get(tier_name, [])

        n_ref = len(ref_ivs)
        total_ref += n_ref

        matched_hyp: set = set()
        matched_ref: set = set()
        tier_added_secs = 0.0
        tier_deleted_secs = 0.0

        # Greedy IoU matching
        for i, (hs, he) in enumerate(hyp_ivs):
            best_iou = 0.0
            best_j = -1
            for j, (rs, re) in enumerate(ref_ivs):
                if j in matched_ref:
                    continue
                overlap = max(0.0, min(he, re) - max(hs, rs))
                union = max(he, re) - min(hs, rs)
                iou = overlap / union if union > 0 else 0.0
                if iou > best_iou:
                    best_iou = iou
                    best_j = j

            if best_iou >= iou_threshold and best_j >= 0:
                matched_hyp.add(i)
                matched_ref.add(best_j)
                rs, re = ref_ivs[best_j]
                # Temporal error for the matched pair
                tier_added_secs += max(0.0, rs - hs) + max(0.0, he - re)   # hyp beyond ref
                tier_deleted_secs += max(0.0, hs - rs) + max(0.0, re - he)  # ref not covered

        # Unmatched hyp → added
        unmatched_hyp = len(hyp_ivs) - len(matched_hyp)
        for i, (hs, he) in enumerate(hyp_ivs):
            if i not in matched_hyp:
                tier_added_secs += he - hs

        # Unmatched ref → deleted
        unmatched_ref = len(ref_ivs) - len(matched_ref)
        for j, (rs, re) in enumerate(ref_ivs):
            if j not in matched_ref:
                tier_deleted_secs += re - rs

        total_added += unmatched_hyp
        total_deleted += unmatched_ref
        total_added_secs += tier_added_secs
        total_deleted_secs += tier_deleted_secs

        per_tier[tier_name] = {
            "n_ref": n_ref,
            "n_hyp": len(hyp_ivs),
            "added_segs": unmatched_hyp,
            "deleted_segs": unmatched_ref,
            "added_secs": round(tier_added_secs, 4),
            "deleted_secs": round(tier_deleted_secs, 4),
        }

    ser = (total_added + total_deleted) / total_ref if total_ref > 0 else 0.0
    ster = (total_added_secs + total_deleted_secs) / total_ref if total_ref > 0 else 0.0

    return {
        "ser": round(ser, 4),
        "ster": round(ster, 4),
        "total_ref_segments": total_ref,
        "total_added_segments": total_added,
        "total_deleted_segments": total_deleted,
        "total_added_secs": round(total_added_secs, 4),
        "total_deleted_secs": round(total_deleted_secs, 4),
        "per_tier": per_tier,
    }


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
            f"boundary_mae={s1.get('boundary_mae_s', '?')}s "
            f"SER={s1.get('ser', '?')} StER={s1.get('ster', '?')}s"
        )
    if s2:
        parts.append(f"Stage2 CER={s2.get('overall_cer', '?'):.2%}")
    if s3:
        parts.append(f"Stage3 CER={s3.get('overall_cer', '?'):.2%}")

    return " | ".join(parts)
