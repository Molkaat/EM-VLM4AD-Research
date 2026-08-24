# EM-VLM4AD Project Summary

## Overview
EM-VLM4AD is a multi-frame vision-language model for question answering in autonomous driving scenes.
It combines:
- A text generation backbone (T5-Base or T5-Large)
- Multi-camera image features (ViT-B/32)
- A gated pooling mechanism to fuse frame/view information

The repository supports end-to-end training, checkpointing, and COCO-style metric evaluation.

## Repository Layout
- `train.py`: Main training entrypoint (custom training loop by default)
- `eval.py`: Inference + COCO caption metric evaluation
- `verify_setup.py`: Validates required dataset/checkpoint files exist
- `modules/multi_frame_dataset.py`: Dataset + collate logic for multi-frame QA samples
- `modules/multi_frame_model.py`: Model architecture (T5 + ViT + gated pooling attention)
- `colab/train_T5_Base.ipynb`: Colab training workflow for T5-Base
- `colab/train_T5_Large.ipynb`: Colab training workflow for T5-Large (LoRA/efficient setup)
- `colab/eval.ipynb`: Colab evaluation workflow and metrics export

## Data Format and Loading
`MultiFrameDataset` loads JSON records where each sample includes:
- A question-answer pair (`Q`, `A`)
- A set of image paths (multi-view/multi-frame)

Processing steps:
1. Build question prompt as: `Question: <Q> Answer:`
2. Read and transform each image
3. Stack images into one tensor per sample
4. Tokenize question and answer for T5 input/labels

The training collate returns:
- Encoded questions
- Image tensor batch
- Encoded labels

The test collate additionally returns raw question text and image paths for prediction bookkeeping.

## Model Architecture
`DriveVLMT5` includes:
- T5 language model (`google-t5/t5-base` or `google-t5/t5-large`)
- ViT-B/32 image encoder from torchvision
- `MultiViewProcessor` that fuses multiple image embeddings using gated pooling attention (GPA)
- Modality embeddings to distinguish text and image tokens before concatenation

Forward path:
1. Encode/fuse image tokens from multiple frames/views
2. Obtain text token embeddings from T5 input embedding table
3. Add modality embeddings to both modalities
4. Concatenate text + image embeddings
5. Feed as `inputs_embeds` into T5 for generation/training loss

For larger LM settings, projection and LoRA adapters are used to improve memory/performance tradeoffs.

## Training Workflow (`train.py`)
Main steps:
1. Parse hyperparameters (LR, epochs, batch size, LM type, LoRA params, checkpoint options)
2. Build train/val/test datasets and dataloaders
3. Initialize `DriveVLMT5`
4. Run custom epoch loop:
   - Forward + loss
   - Backprop + optimizer step
   - Validation at epoch end
   - LR scheduling
   - Checkpoint/stat JSON saving
5. Save loss plot and experiment CSV summary
6. Evaluate best saved checkpoint on test split (loss)

Saved artifacts are written under `multi_frame_results/<run_or_checkpoint_name>/`.

## Evaluation Workflow (`eval.py`)
Main steps:
1. Load model checkpoint from `multi_frame_results/<model_name>/latest_model.pth`
2. Run generation over the test split
3. Save predictions as `predictions.json`
4. Compute COCO-style metrics using:
   - `pycocotools`
   - `pycocoevalcap`
5. Save metric table to `metrics.csv`

Metrics include BLEU, CIDEr, METEOR, and ROUGE-L (SPICE may be slow/disabled in practice).

## Setup Validation (`verify_setup.py`)
Checks whether expected files exist, including:
- Dataset split JSON files
- COCO evaluation JSON
- image_id mapping
- Pretrained checkpoints for T5-Medium/T5-Large folders

If missing, it prints guidance for downloading and placing files correctly.

## Colab Notebooks Summary
### `colab/train_T5_Base.ipynb`
- Installs required packages in Colab
- Mounts Google Drive
- Defines model/dataset/training code in-notebook
- Uses T5-Base configuration
- Trains, validates, and saves checkpoints/stats/plots/results CSV to Drive

### `colab/train_T5_Large.ipynb`
- Same flow as Base notebook, but for T5-Large
- Installs additional efficiency-related packages (`accelerate`, `bitsandbytes`)
- Uses LoRA-oriented configuration for practical large-model fine-tuning

### `colab/eval.ipynb`
- Loads a trained checkpoint from Drive
- Generates answers on test data
- Saves prediction JSON
- Computes and exports COCO metrics CSV

## Typical Commands
```bash
python verify_setup.py
python train.py --batch-size 4 --epochs 15 --lm T5-Base
python train.py --batch-size 4 --epochs 15 --lm T5-Large --lora
python eval.py --batch-size 4 --lm T5-Base --model-name T5-Medium
```

## Key Dependencies
From `requirements.txt`:
- `transformers`
- `peft`
- `accelerate`
- `bitsandbytes`
- `pycocotools`
- `pycocoevalcap`

## Notes
- The code uses CUDA when available.
- Results/checkpoints are organized under `multi_frame_results`.
- Google Colab notebooks are self-contained versions of local train/eval scripts for Drive-based workflows.
