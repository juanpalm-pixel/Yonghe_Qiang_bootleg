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

### Action Required Before Running Stage 1
To enable the pyannote diarization model, you must:
1. Accept the model terms at: https://huggingface.co/pyannote/speaker-diarization-3.1
2. Accept segmentation model terms at: https://huggingface.co/pyannote/segmentation-3.0
3. In HuggingFace settings, enable "Access to public gated repositories" for your fine-grained token.

### Action Required for GitHub Push
1. Create the repository: https://github.com/new → name: `Yonghe_Qiang_bootleg`
2. In the repo directory run:
   ```
   git remote add origin https://github.com/juanpalm-pixel/Yonghe_Qiang_bootleg.git
   git push -u origin main
   ```

---
## [trial_1] 2026-03-06 11:27

### Summary
```
Stage1 coverage=?% boundary_mae=?s | Stage2 CER=100.00%
```

### Stage 1 — Diarization
- Speech coverage: ? %
- Boundary MAE: ? s
- Hypothesis segments: ?
- Reference segments: ?

### Stage 2 — IPA (no tones)
- Overall CER: 100.00%
  - Tier A: CER=100.00%  ref=1030 chars  hyp=0 chars
  - Tier B: CER=100.00%  ref=1084 chars  hyp=0 chars
  - Tier C: CER=100.00%  ref=95 chars  hyp=0 chars

### Stage 3 — IPA with tones

### Errors / Issues
- Stage 1 FAILED: Pipeline.from_pretrained() got an unexpected keyword argument 'use_auth_token'
- Stage 3 FAILED: Could not load libtorchcodec. Likely causes:
          1. FFmpeg is not properly installed in your environment. We support
             versions 4, 5, 6, 7, and 8, and we attempt to load libtorchcodec
             for each of those versions. Errors for versions not installed on
             your system are expected; only the error for your installed FFmpeg
             version is relevant. On Windows, ensure you've installed the
             "full-shared" version which ships DLLs.
          2. The PyTorch version (2.10.0+cpu) is not compatible with
             this version of TorchCodec. Refer to the version compatibility
             table:
             https://github.com/pytorch/torchcodec?tab=readme-ov-file#installing-torchcodec.
          3. Another runtime dependency; see exceptions below.

        The following exceptions were raised as we tried to load libtorchcodec:
        
[start of libtorchcodec loading traceback]
FFmpeg version 8:
Traceback (most recent call last):
  File "C:\Users\pablo\miniconda3\envs\yonghe_qiang\Lib\site-packages\torch\_ops.py", line 1442, in load_library
    ctypes.CDLL(path)
  File "C:\Users\pablo\miniconda3\envs\yonghe_qiang\Lib\ctypes\__init__.py", line 376, in __init__
    self._handle = _dlopen(self._name, mode)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: Could not find module 'C:\Users\pablo\miniconda3\envs\yonghe_qiang\Lib\site-packages\torchcodec\libtorchcodec_core8.dll' (or one of its dependencies). Try using the full path with constructor syntax.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "C:\Users\pablo\miniconda3\envs\yonghe_qiang\Lib\site-packages\torchcodec\_core\ops.py", line 57, in load_torchcodec_shared_libraries
    torch.ops.load_library(core_library_path)
  File "C:\Users\pablo\miniconda3\envs\yonghe_qiang\Lib\site-packages\torch\_ops.py", line 1444, in load_library
    raise OSError(f"Could not load this library: {path}") from e
OSError: Could not load this library: C:\Users\pablo\miniconda3\envs\yonghe_qiang\Lib\site-packages\torchcodec\libtorchcodec_core8.dll

FFmpeg version 7:
Traceback (most recent call last):
  File "C:\Users\pablo\miniconda3\envs\yonghe_qiang\Lib\site-packages\torch\_ops.py", line 1442, in load_library
    ctypes.CDLL(path)
  File "C:\Users\pablo\miniconda3\envs\yonghe_qiang\Lib\ctypes\__init__.py", line 376, in __init__
    self._handle = _dlopen(self._name, mode)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: Could not find module 'C:\Users\pablo\miniconda3\envs\yonghe_qiang\Lib\site-packages\torchcodec\libtorchcodec_core7.dll' (or one of its dependencies). Try using the full path with constructor syntax.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "C:\Users\pablo\miniconda3\envs\yonghe_qiang\Lib\site-packages\torchcodec\_core\ops.py", line 57, in load_torchcodec_shared_libraries
    torch.ops.load_library(core_library_path)
  File "C:\Users\pablo\miniconda3\envs\yonghe_qiang\Lib\site-packages\torch\_ops.py", line 1444, in load_library
    raise OSError(f"Could not load this library: {path}") from e
OSError: Could not load this library: C:\Users\pablo\miniconda3\envs\yonghe_qiang\Lib\site-packages\torchcodec\libtorchcodec_core7.dll

FFmpeg version 6:
Traceback (most recent call last):
  File "C:\Users\pablo\miniconda3\envs\yonghe_qiang\Lib\site-packages\torch\_ops.py", line 1442, in load_library
    ctypes.CDLL(path)
  File "C:\Users\pablo\miniconda3\envs\yonghe_qiang\Lib\ctypes\__init__.py", line 376, in __init__
    self._handle = _dlopen(self._name, mode)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: Could not find module 'C:\Users\pablo\miniconda3\envs\yonghe_qiang\Lib\site-packages\torchcodec\libtorchcodec_core6.dll' (or one of its dependencies). Try using the full path with constructor syntax.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "C:\Users\pablo\miniconda3\envs\yonghe_qiang\Lib\site-packages\torchcodec\_core\ops.py", line 57, in load_torchcodec_shared_libraries
    torch.ops.load_library(core_library_path)
  File "C:\Users\pablo\miniconda3\envs\yonghe_qiang\Lib\site-packages\torch\_ops.py", line 1444, in load_library
    raise OSError(f"Could not load this library: {path}") from e
OSError: Could not load this library: C:\Users\pablo\miniconda3\envs\yonghe_qiang\Lib\site-packages\torchcodec\libtorchcodec_core6.dll

FFmpeg version 5:
Traceback (most recent call last):
  File "C:\Users\pablo\miniconda3\envs\yonghe_qiang\Lib\site-packages\torch\_ops.py", line 1442, in load_library
    ctypes.CDLL(path)
  File "C:\Users\pablo\miniconda3\envs\yonghe_qiang\Lib\ctypes\__init__.py", line 376, in __init__
    self._handle = _dlopen(self._name, mode)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: Could not find module 'C:\Users\pablo\miniconda3\envs\yonghe_qiang\Lib\site-packages\torchcodec\libtorchcodec_core5.dll' (or one of its dependencies). Try using the full path with constructor syntax.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "C:\Users\pablo\miniconda3\envs\yonghe_qiang\Lib\site-packages\torchcodec\_core\ops.py", line 57, in load_torchcodec_shared_libraries
    torch.ops.load_library(core_library_path)
  File "C:\Users\pablo\miniconda3\envs\yonghe_qiang\Lib\site-packages\torch\_ops.py", line 1444, in load_library
    raise OSError(f"Could not load this library: {path}") from e
OSError: Could not load this library: C:\Users\pablo\miniconda3\envs\yonghe_qiang\Lib\site-packages\torchcodec\libtorchcodec_core5.dll

FFmpeg version 4:
Traceback (most recent call last):
  File "C:\Users\pablo\miniconda3\envs\yonghe_qiang\Lib\site-packages\torch\_ops.py", line 1442, in load_library
    ctypes.CDLL(path)
  File "C:\Users\pablo\miniconda3\envs\yonghe_qiang\Lib\ctypes\__init__.py", line 376, in __init__
    self._handle = _dlopen(self._name, mode)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: Could not find module 'C:\Users\pablo\miniconda3\envs\yonghe_qiang\Lib\site-packages\torchcodec\libtorchcodec_core4.dll' (or one of its dependencies). Try using the full path with constructor syntax.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "C:\Users\pablo\miniconda3\envs\yonghe_qiang\Lib\site-packages\torchcodec\_core\ops.py", line 57, in load_torchcodec_shared_libraries
    torch.ops.load_library(core_library_path)
  File "C:\Users\pablo\miniconda3\envs\yonghe_qiang\Lib\site-packages\torch\_ops.py", line 1444, in load_library
    raise OSError(f"Could not load this library: {path}") from e
OSError: Could not load this library: C:\Users\pablo\miniconda3\envs\yonghe_qiang\Lib\site-packages\torchcodec\libtorchcodec_core4.dll
[end of libtorchcodec loading traceback].
- Stage 3 eval FAILED: cannot access local variable 'step3_tg' where it is not associated with a value

---
## [trial_2] 2026-03-06 11:31

### Summary
```
Stage1 coverage=?% boundary_mae=?s | Stage2 CER=100.00% | Stage3 CER=100.00%
```

### Stage 1 — Diarization
- Speech coverage: ? %
- Boundary MAE: ? s
- Hypothesis segments: ?
- Reference segments: ?

### Stage 2 — IPA (no tones)
- Overall CER: 100.00%
  - Tier A: CER=100.00%  ref=1030 chars  hyp=0 chars
  - Tier B: CER=100.00%  ref=1084 chars  hyp=0 chars
  - Tier C: CER=100.00%  ref=95 chars  hyp=0 chars

### Stage 3 — IPA with tones
- Overall CER: 100.00%
  - Tier A: CER=100.00%  ref=1838 chars  hyp=0 chars
  - Tier B: CER=100.00%  ref=1960 chars  hyp=0 chars
  - Tier C: CER=100.00%  ref=167 chars  hyp=0 chars

### Errors / Issues
- Stage 1 FAILED: An error happened while trying to locate the file on the Hub and we cannot find the requested files in the local cache. Please check your connection and try again or make sure your Internet connection is on.

---
## [trial_3] 2026-03-06 12:20

### Summary
```
Stage1 coverage=?% boundary_mae=?s | Stage2 CER=388.73% | Stage3 CER=263.08%
```

### Stage 1 — Diarization
- Speech coverage: ? %
- Boundary MAE: ? s
- Hypothesis segments: ?
- Reference segments: ?

### Stage 2 — IPA (no tones)
- Overall CER: 388.73%
  - Tier A: CER=280.68%  ref=1030 chars  hyp=3285 chars
  - Tier B: CER=273.15%  ref=1084 chars  hyp=3373 chars
  - Tier C: CER=2878.95%  ref=95 chars  hyp=2804 chars

### Stage 3 — IPA with tones
- Overall CER: 263.08%
  - Tier A: CER=183.19%  ref=1838 chars  hyp=4229 chars
  - Tier B: CER=177.70%  ref=1960 chars  hyp=4331 chars
  - Tier C: CER=2144.31%  ref=167 chars  hyp=3710 chars

### Errors / Issues
- Stage 1 FAILED: An error happened while trying to locate the file on the Hub and we cannot find the requested files in the local cache. Please check your connection and try again or make sure your Internet connection is on.

---
## [trial_4] 2026-03-06 13:32

### Summary
```
Stage1 coverage=?% boundary_mae=?s SER=0.0 StER=0.0s | Stage2 CER=128.47% | Stage3 CER=90.49%
```

### Stage 1 — Diarization
- Speech coverage: ? %
- Boundary MAE: ? s
- Hypothesis segments: ?
- Reference segments: ?
- SER: 0.0 (added=0, deleted=0)
- StER: 0.0 s/seg (added=0.0s, deleted=0.0s)

### Stage 2 — IPA (no tones)
- Overall CER: 128.47%
  - Tier A: CER=122.82%  ref=1030 chars  hyp=1503 chars
  - Tier B: CER=136.44%  ref=1084 chars  hyp=1777 chars
  - Tier C: CER=98.95%  ref=95 chars  hyp=3 chars

### Stage 3 — IPA with tones
- Overall CER: 90.49%
  - Tier A: CER=86.34%  ref=1838 chars  hyp=1907 chars
  - Tier B: CER=93.72%  ref=1960 chars  hyp=2249 chars
  - Tier C: CER=98.20%  ref=167 chars  hyp=5 chars

### Errors / Issues
- Stage 1 FAILED: 403 Client Error. (Request ID: Root=1-69aad6e7-45500f3166a329ef38bc8145;4b9e619d-14f7-4fe2-92e2-5bc665986aa9)

Cannot access gated repo for url https://huggingface.co/pyannote/speaker-diarization-community-1/resolve/main/plda/xvec_transform.npz.
Access to model pyannote/speaker-diari…

---
## [trial_stage1_rerun] 2026-03-06 13:58

### Summary
```
Stage1 coverage=?% boundary_mae=?s SER=0.0 StER=0.0s | Stage2 CER=128.47% | Stage3 CER=90.49%
```

### Stage 1 — Diarization
- Speech coverage: ? %
- Boundary MAE: ? s
- Hypothesis segments: ?
- Reference segments: ?
- SER: 0.0 (added=0, deleted=0)
- StER: 0.0 s/seg (added=0.0s, deleted=0.0s)

### Stage 2 — IPA (no tones)
- Overall CER: 128.47%
  - Tier A: CER=122.82%  ref=1030 chars  hyp=1503 chars
  - Tier B: CER=136.44%  ref=1084 chars  hyp=1777 chars
  - Tier C: CER=98.95%  ref=95 chars  hyp=3 chars

### Stage 3 — IPA with tones
- Overall CER: 90.49%
  - Tier A: CER=86.34%  ref=1838 chars  hyp=1907 chars
  - Tier B: CER=93.72%  ref=1960 chars  hyp=2249 chars
  - Tier C: CER=98.20%  ref=167 chars  hyp=5 chars

### Errors / Issues
- Stage 1 FAILED: 'DiarizeOutput' object has no attribute 'itertracks'

---
## [trial_stage1_rerun_2] 2026-03-06 14:04

### Summary
```
Stage1 coverage=89.75% boundary_mae=0.1288s SER=1.0104 StER=1.1581s | Stage2 CER=118.33% | Stage3 CER=85.88%
```

### Stage 1 — Diarization
- Speech coverage: 89.75 %
- Boundary MAE: 0.1288 s
- Hypothesis segments: 88
- Reference segments: 193
- SER: 1.0104 (added=45, deleted=150)
- StER: 1.1581 s/seg (added=111.2399s, deleted=112.2774s)

### Stage 2 — IPA (no tones)
- Overall CER: 118.33%
  - Tier A: CER=127.28%  ref=1030 chars  hyp=1548 chars
  - Tier B: CER=105.35%  ref=1084 chars  hyp=1320 chars
  - Tier C: CER=169.47%  ref=95 chars  hyp=183 chars

### Stage 3 — IPA with tones
- Overall CER: 85.88%
  - Tier A: CER=88.79%  ref=1838 chars  hyp=1992 chars
  - Tier B: CER=80.82%  ref=1960 chars  hyp=1706 chars
  - Tier C: CER=113.17%  ref=167 chars  hyp=233 chars
