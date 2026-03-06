# Changelog — Yonghe Qiang ASR Bootleg

All significant changes, trial runs, errors, decisions, and CER results are documented here.

---

## [Setup] Initial project creation — 2026-03-06

### Environment
- Conda environment `yonghe_qiang` (Python 3.11)
- PyTorch 2.10.0+cpu (CPU-only; no NVIDIA GPU available — graceful CPU fallback implemented)
- Key packages: transformers 5.3.0, pyannote.audio 4.0.4, librosa 0.11.0, praatio 6.2.2, jiwer 4.0.0

### Model choice
- **Base model**: `YonaKhine/finetuned-w2v2-bert-burmese-asr`
  - Architecture: `facebook/w2v-bert-2.0` fine-tuned on OpenSLR-80 Burmese speech
  - Rationale: Burmese (Tibeto-Burman) is the closest available language with a Wav2Vec2 ASR model to Yonghe Qiang (also Tibeto-Burman, Qiangic branch)
  - WER on Burmese eval set: ~42.6 % (fine-tuned result)
- **Diarization model**: `pyannote/speaker-diarization-3.1`
- **Tone prediction**: `librosa.pyin` F0 estimator + rule-based pitch classifier

### Known Issues at Setup
1. **GH_TOKEN insufficient permissions**: The `GH_TOKEN` environment variable points to a fine-grained PAT
   without `Repositories: Write` permission. Automatic GitHub repo creation via API returns HTTP 403.
   **Workaround**: Manually create the repository at https://github.com/new → name: `Yonghe_Qiang_bootleg`.
   Then run `git remote add origin https://github.com/juanpalm-pixel/Yonghe_Qiang_bootleg.git`.
   The `run_pipeline.py` script will detect the remote and push automatically if it exists.

### Design Decisions
- **Stage 1 (Diarization)**: pyannote produces speaker clusters labelled `SPEAKER_00`, `SPEAKER_01`, etc.
  These are mapped to A/B/C tiers by matching the largest temporal overlap with reference tier timestamps.
  DER (Diarization Error Rate) and segment boundary mean absolute error (seconds) are logged instead of
  CER for this stage since the output text is empty.
- **Stage 2 (IPA)**: Burmese Unicode output from the model is mapped character-by-character (and by common
  digraphs) to IPA using a handcrafted mapping table (`src/burmese_ipa_map.py`). The mapping targets the
  phoneme inventory documented for Yonghe Qiang. Tone numbers are stripped in this stage.
- **Stage 3 (Tones)**: Fundamental frequency (F0) extracted with `librosa.pyin` (probabilistic YIN).
  Per-interval mean F0 and normalised slope are used to classify into Yonghe Qiang tone categories:
  - **55** — high level (high mean F0, near-zero slope)
  - **53** — high falling (high mean F0, negative slope)
  - **35** — low rising (low mean F0, positive slope)
  - **33** — mid level (mid mean F0, near-zero slope)
  Syllable segmentation uses a simple onset-detector over the IPA string (each consonant cluster + vowel
  nucleus = one syllable). The tone number is appended after each IPA syllable.

---

<!-- Trial entries will be appended here automatically by run_pipeline.py -->
