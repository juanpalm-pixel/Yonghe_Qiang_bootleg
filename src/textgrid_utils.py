"""
textgrid_utils.py
-----------------
Read and write Praat TextGrid files using praatio.

The TextGrid format used in this project has 3 IntervalTiers (A, B, C)
representing the three speakers in the Yonghe Qiang recording.
"""

from __future__ import annotations

import re
from typing import List, Tuple

from praatio import textgrid as tgio
from praatio.data_classes.interval_tier import Interval


def read_textgrid(path: str) -> tgio.Textgrid:
    """Load a TextGrid from *path* and return a praatio Textgrid object."""
    return tgio.openTextgrid(path, includeEmptyIntervals=True)


def write_textgrid(tg: tgio.Textgrid, path: str) -> None:
    """Save *tg* to *path* in Praat short format (UTF-8)."""
    tg.save(path, format="short_textgrid", includeBlankSpaces=True)


def get_intervals(tg: tgio.Textgrid, tier_name: str) -> List[Tuple[float, float, str]]:
    """
    Return all intervals for *tier_name* as a list of (start, end, text) tuples.
    Empty-text intervals are included.
    """
    tier = tg.getTier(tier_name)
    return [(iv.start, iv.end, iv.label) for iv in tier.entries]


def get_nonempty_intervals(
    tg: tgio.Textgrid, tier_name: str
) -> List[Tuple[float, float, str]]:
    """Return only intervals with non-empty text."""
    return [(s, e, t) for s, e, t in get_intervals(tg, tier_name) if t.strip()]


def build_textgrid(
    duration: float,
    tier_data: dict,  # {tier_name: [(start, end, text), ...]}
) -> tgio.Textgrid:
    """
    Construct a new Textgrid with the given tier data.

    Parameters
    ----------
    duration : float
        Total duration of the audio in seconds.
    tier_data : dict
        Mapping from tier name to list of (start, end, text) tuples.
        All intervals covering 0..duration must be provided (no gaps allowed).
    """
    tg = tgio.Textgrid()
    tg.minTimestamp = 0.0
    tg.maxTimestamp = duration

    for name, intervals in tier_data.items():
        entries = [Interval(s, e, t) for s, e, t in intervals]
        tier = tgio.IntervalTier(name, entries, 0.0, duration)
        tg.addTier(tier)

    return tg


def copy_structure_empty(source_tg: tgio.Textgrid) -> tgio.Textgrid:
    """
    Clone the structure of *source_tg* (tier names + interval boundaries)
    but set all text labels to empty strings.
    """
    tier_data = {}
    for name in source_tg.tierNames:
        intervals = [(s, e, "") for s, e, _ in get_intervals(source_tg, name)]
        tier_data[name] = intervals
    return build_textgrid(source_tg.maxTimestamp, tier_data)


def strip_tone_numbers(text: str) -> str:
    """Remove numeric tone suffixes (e.g. '33', '55', '53', '35') from IPA text."""
    return re.sub(r"\b\d{2}\b", "", text).strip()


def extract_all_text(tg: tgio.Textgrid) -> List[str]:
    """Collect all non-empty text labels across all tiers, in time order."""
    entries = []
    for name in tg.tierNames:
        for s, e, t in get_intervals(tg, name):
            if t.strip():
                entries.append((s, t))
    entries.sort(key=lambda x: x[0])
    return [t for _, t in entries]
