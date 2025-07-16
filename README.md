# Multilingual ASR Evaluation Pipeline (Amharic/Oromo)

This project implements a comprehensive evaluation pipeline for multilingual Automatic Speech Recognition (ASR) models, specifically focused on African languages such as **Amharic** and **Oromo**. It leverages Hugging Face’s `transformers` library, datasets from the Hub, and standardized evaluation metrics including **Word Error Rate (WER)** and **Character Error Rate (CER)**.

## ✅ Features

- Supports batch inference for large-scale speech datasets
- Evaluates pre-trained Whisper/MMS models from Hugging Face Hub
- Computes per-sample and overall WER/CER
- Saves detailed transcription results and summary CSV reports
- Handles dataset filtering, preprocessing, and duration control
- Compatible with GPU acceleration (`pipeline(..., device=0)`)

## 📁 Datasets

Preloaded datasets used for evaluation include:
- `KYAGABA/amharic_cleaned_testset_verified`
- `KYAGABA/amharic_cleaned_testset_common_voice_new`
- `KYAGABA/AMHARIC_ALFA_DATASET_new_current`
- `KYAGABA/fleurs_voice_luo_new`

All datasets are filtered for audio duration, cleaned transcription text, and ratio of audio-to-text length.

## 🧪 Models

Evaluate models such as:
- `asr-africa/whisper-small-amharic_dataset-ormo-100hrs-v6`
- `asr-africa/facebook-mms-1b-all-common_voice_fleurs-amh-200hrs-v1`

Supports both Whisper and MMS architectures via Hugging Face pipelines.

## ⚙️ Requirements

```bash
pip install datasets transformers evaluate jiwer torchaudio librosa huggingface_hub accelerate
