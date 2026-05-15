# GTZAN Music Genre Recognition

Music genre recognition project built on the GTZAN dataset. The repository contains exploratory analysis, traditional machine learning baselines, a CNN-based audio classifier trained on Mel-spectrograms, and a Streamlit demo that serves an exported ONNX model.

## Project overview

The goal of the project is to classify 30-second music clips into 10 genres:

blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock.

The main workflow is:

1. Explore the dataset and engineered audio features.
2. Train classic ML baselines on tabular features.
3. Train a CNN on Mel-spectrogram images.
4. Export the best CNN to ONNX.
5. Run inference through a Streamlit app.

## Repository structure

```text
data/
	features_3_sec.csv
	features_30_sec.csv
	genres_original/
	images_original/
Data/
	spectrograms/
	spectrograms_segmented/
deploy/
	04_Streamlit_App.py
models/
	cnn_best.pt
	cnn_gtzan.onnx
music/
notebooks/
	01_EDA_with_theory.ipynb
	01_EDA.ipynb
	02_Baseline_ML.ipynb
	03_CNN_Model.ipynb
	03_CNN_Model_v2.ipynb
	03_CNN_Model_v2_colab.ipynb
```

## Main components

### Data

- `data/genres_original/` contains the GTZAN audio clips organized by genre.
- `data/features_30_sec.csv` and `data/features_3_sec.csv` provide precomputed features for classical ML experiments.
- `Data/spectrograms_segmented/` stores the generated Mel-spectrogram images used by the CNN pipeline.

### Notebooks

- `01_EDA_with_theory.ipynb` and `01_EDA.ipynb` cover exploratory analysis and feature inspection.
- `02_Baseline_ML.ipynb` covers classical baselines such as logistic regression, random forest, and SVC.
- `03_CNN_Model.ipynb`, `03_CNN_Model_v2.ipynb`, and `03_CNN_Model_v2_colab.ipynb` cover the CNN workflow on Mel-spectrograms.

### Deployment

- `deploy/04_Streamlit_App.py` is the app entrypoint.
- `models/cnn_gtzan.onnx` is the model used by the Streamlit app.

## Results summary

The best classical baseline in the repository is the random forest model trained on tabular features.

The CNN pipeline operates on Mel-spectrograms and was later exported to ONNX for deployment. The Streamlit demo loads the ONNX model, accepts uploaded audio files, splits them into segments, generates Mel-spectrograms, and shows the predicted genre distribution.

## Current progress (2026-05-15)

- Data acquisition and organization completed; raw audio available under `music/` and original labels under `data/genres_original/`.
- Preprocessing and feature extraction completed; generated artifacts include `features_3_sec.csv`, `features_30_sec.csv`, and spectrogram images in `Data/spectrograms/` and `Data/spectrograms_segmented/`.
- Exploratory Data Analysis performed (`notebooks/01_EDA_with_theory.ipynb`).
- Baseline ML experiments executed (`notebooks/02_Baseline_ML.ipynb`).
- CNN training completed with best model saved to `models/cnn_best.pt` and exported to `models/cnn_gtzan.onnx` for deployment.
- Streamlit demo prepared at `deploy/04_Streamlit_App.py` to run inference using the ONNX model.


## Environment setup

Use the provided Conda environment file:

```bash
conda env create -f gtzan_env.yaml
conda activate gtzan
```

If you only want to run the Streamlit app, make sure the environment includes:

- `streamlit`
- `onnxruntime-gpu` or `onnxruntime`
- `librosa`
- `numpy`
- `matplotlib`
- `Pillow`

## Run the Streamlit app

From the project root:

```bash
streamlit run deploy/04_Streamlit_App.py
```

The app expects `models/cnn_gtzan.onnx` to be present. If the file is missing, the app will stop with an error message.

## Reproducibility notes

- The dataset is the 10-class GTZAN music genre dataset.
- Audio is converted to Mel-spectrograms with `librosa`.
- The deployed model is a CNN exported to ONNX.
- The repository keeps both research artifacts and deployment artifacts so the full pipeline can be inspected end to end.

## License

See `LICENSE` for the project license.