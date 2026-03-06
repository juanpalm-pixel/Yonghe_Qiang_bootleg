# Yonghe Qiang ASR Bootleg

Zero-shot ASR pipeline for **Yonghe Qiang** (Glottolog: `yong1287`) bootstrapped from a Burmese Wav2Vec2-BERT model.

## Language
- **Target language**: Yonghe Qiang (Qiangic, Tibeto-Burman)
- **Donor model**: `YonaKhine/finetuned-w2v2-bert-burmese-asr` (Burmese — also Tibeto-Burman)

## Pipeline Stages

| Stage | Description | Benchmark |
|-------|-------------|-----------|
| 1 | Speaker diarization (A / B / C) | `YH-758_1_IPA-SEPERATE-EMPTY.TextGrid` |
| 2 | IPA transcription without tones | `YH-758_2_IPA-SEPERATE-NO_TONES.TextGrid` |
| 3 | Tone prediction & insertion | `YH-758_3_IPA-SEPERATE-WITH_TONES.TextGrid` |

## Quick Start

```bash
# Activate conda environment
conda activate yonghe_qiang

# Run full pipeline (creates outputs/trial_N/)
python run_pipeline.py \
    --audio "C:/Users/pablo/OneDrive/Desktop/Functions/yonghe-qian_01/YH-758 - Benchmark/YH-758.wav" \
    --benchmark-dir "C:/Users/pablo/OneDrive/Desktop/Functions/yonghe-qian_01/YH-758 - Benchmark"
```

## Environment

```bash
conda env create -f environment.yml
```

## Results Log

See [CHANGELOG.md](CHANGELOG.md) for full trial history with CER scores.

## Repository Structure

```
src/
  diarize.py           # Stage 1: pyannote speaker diarization
  transcribe_ipa.py    # Stage 2: Burmese model + IPA mapping
  tone_predict.py      # Stage 3: F0-based tone classification
  evaluate.py          # CER / DER evaluation
  textgrid_utils.py    # TextGrid read/write helpers
  burmese_ipa_map.py   # Burmese → IPA phoneme mapping table
run_pipeline.py        # Orchestrator
outputs/
  trial_N/             # Per-trial TextGrids and CER report
CHANGELOG.md           # Development log
```
