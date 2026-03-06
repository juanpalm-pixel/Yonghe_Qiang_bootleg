"""
transcribe_ipa.py
-----------------
Stage 2: IPA transcription without tones.

Uses the Burmese Wav2Vec2-BERT model (YonaKhine/finetuned-w2v2-bert-burmese-asr)
to extract acoustic features and produce a rough transcription, then maps
the Burmese script output to approximate IPA using burmese_ipa_map.py.

Zero-shot transfer rationale
-----------------------------
Burmese (Lolo-Burmese) and Yonghe Qiang (Qiangic) are both Tibeto-Burman
languages. While their specific phoneme inventories differ, the Burmese
acoustic model has learned robust sub-word representations of sounds that
also occur in Yonghe Qiang, making it a useful bootstrapping point.

Evaluation target: YH-758_2_IPA-SEPERATE-NO_TONES.TextGrid
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torchaudio

logger = logging.getLogger(__name__)

MODEL_NAME = "YonaKhine/finetuned-w2v2-bert-burmese-asr"
TARGET_SR = 16_000  # model expects 16 kHz


def get_device() -> str:
    if torch.cuda.is_available():
        logger.info("CUDA GPU detected — using GPU for transcription.")
        return "cuda"
    logger.info("No CUDA GPU — falling back to CPU for transcription.")
    return "cpu"


def load_model(hf_token: str | None = None):
    """
    Load the Burmese Wav2Vec2-BERT model and processor.

    Returns (processor, model, device).
    """
    from transformers import AutoProcessor, AutoModelForCTC

    device = get_device()
    logger.info("Loading model %s …", MODEL_NAME)

    kwargs = {}
    if hf_token:
        kwargs["token"] = hf_token

    processor = AutoProcessor.from_pretrained(MODEL_NAME, **kwargs)
    model = AutoModelForCTC.from_pretrained(MODEL_NAME, **kwargs)
    model = model.to(device)
    model.eval()

    return processor, model, device


def extract_audio_segment(
    audio_path: str,
    start_sec: float,
    end_sec: float,
    target_sr: int = TARGET_SR,
) -> np.ndarray:
    """
    Load *audio_path*, extract the segment [start_sec, end_sec] using soundfile
    (avoids torchcodec DLL issues on Windows), and return a float32 numpy array.
    """
    import soundfile as sf

    info = sf.info(audio_path)
    sr = info.samplerate
    total_frames = info.frames

    start_frame = int(start_sec * sr)
    end_frame = min(int(end_sec * sr), total_frames)
    n_frames = end_frame - start_frame

    if n_frames <= 0:
        return np.array([], dtype=np.float32)

    audio_data, _ = sf.read(
        audio_path,
        start=start_frame,
        frames=n_frames,
        dtype="float32",
        always_2d=True,
    )

    # Mono
    if audio_data.shape[1] > 1:
        mono = audio_data.mean(axis=1)
    else:
        mono = audio_data[:, 0]

    # Resample if necessary
    if sr != target_sr:
        import librosa
        mono = librosa.resample(mono, orig_sr=sr, target_sr=target_sr)

    return mono.astype(np.float32)


def transcribe_segment(
    audio_array: np.ndarray,
    processor,
    model,
    device: str,
    sample_rate: int = TARGET_SR,
) -> str:
    """
    Run the Burmese ASR model on *audio_array* (float32, 1-D, 16 kHz).
    Returns the raw Burmese Unicode transcription string.
    """
    if len(audio_array) == 0:
        return ""

    inputs = processor(
        audio_array,
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=True,
    )
    input_values = inputs.input_features.to(device) if hasattr(inputs, "input_features") else inputs.input_values.to(device)

    with torch.no_grad():
        logits = model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription


def burmese_output_to_ipa(burmese_text: str) -> str:
    """
    Convert raw Burmese model output → approximate IPA (no tones).
    """
    from burmese_ipa_map import burmese_to_ipa
    return burmese_to_ipa(burmese_text)


def stage2_transcribe(
    audio_path: str,
    tier_data: Dict[str, List[Tuple[float, float, str]]],
    hf_token: str | None = None,
) -> Dict[str, List[Tuple[float, float, str]]]:
    """
    Full Stage 2 pipeline.

    Parameters
    ----------
    audio_path : str
        Path to the WAV file.
    tier_data : dict
        Output from Stage 1: {tier_name: [(start, end, ""), ...]}
    hf_token : str | None
        HuggingFace token (optional).

    Returns
    -------
    dict
        {tier_name: [(start, end, ipa_text), ...]} — same boundaries as
        Stage 1, but non-silent intervals now carry approximate IPA text.
    """
    logger.info("=== Stage 2: IPA Transcription (no tones) ===")

    processor, model, device = load_model(hf_token)

    result: Dict[str, List[Tuple[float, float, str]]] = {}

    for tier_name, intervals in tier_data.items():
        new_intervals = []
        for start, end, _ in intervals:
            duration = end - start
            # Skip very short or silent intervals
            if duration < 0.1:
                new_intervals.append((start, end, ""))
                continue

            audio = extract_audio_segment(audio_path, start, end)

            # Skip if too short (< 0.1 s after extraction)
            if len(audio) < int(0.1 * TARGET_SR):
                new_intervals.append((start, end, ""))
                continue

            burmese_text = transcribe_segment(audio, processor, model, device)
            ipa_text = burmese_output_to_ipa(burmese_text)

            logger.debug(
                "[%s] %.2f–%.2f  Burmese=%r  IPA=%r",
                tier_name, start, end, burmese_text, ipa_text,
            )
            new_intervals.append((start, end, ipa_text))

        result[tier_name] = new_intervals

    return result
