# Brain Tumor Segmentation (LGG MRI)

In this project, we develop a semantic segmentation model to identify and delineate lower-grade glioma (LGG) tumors in brain MRI images using deep learning models.

## Structure
- `src/` : source code
- `outputs/` : predictions, figures, checkpoints

## Dataset

This project uses the LGG MRI Segmentation dataset from Kaggle.

Due to size constraints, the dataset is not included in this repository.

### Download

Download the dataset from:
https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation

### Setup

1. Extract the dataset.
2. Place the extracted folder named "kaggle_3m" inside: ## Dataset

This project uses the LGG MRI Segmentation dataset from Kaggle.

Due to size constraints, the dataset is not included in this repository.

### Download

Download the dataset from:
https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation

### Setup

1. Extract the dataset.
2. Place the extracted folder inside: `data/raw/`

So the structure looks like: `data/raw/kaggle_3m/<patient_folder>/...`

### Notes

- Do not modify the contents of `data/raw/`
- Preprocessed data will be stored in `data/processed/`

## Requirements

Install dependencies:

pip install -r requirements.txt

### GPU Support 

You can enable GPU acceleration by installing a CUDA-enabled version of PyTorch:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128