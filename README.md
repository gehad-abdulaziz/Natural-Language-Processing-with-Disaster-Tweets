# NLP with Disaster Tweets

A tweet classification project built for the [Kaggle NLP Getting Started competition](https://www.kaggle.com/c/nlp-getting-started). The goal is to predict whether a given tweet is about a real disaster or not.

## Overview

Twitter has become an important communication channel during emergencies. The problem is that not every tweet that *sounds* like it's about a disaster actually is — "the fire in my belly" means something very different from "fire in downtown LA." This project trains models to tell the difference.

The dataset contains ~7,600 labeled tweets, each with a `text` field and a binary `target` (1 = real disaster, 0 = not). Some tweets also include optional `keyword` and `location` fields, though these were dropped during preprocessing since the raw text carries most of the signal.

## Approach

Two models were trained and compared:

**Baseline — TF-IDF + Logistic Regression**  
A simple sklearn pipeline with TF-IDF vectorization (English stopwords removed, top 10k features) followed by Logistic Regression. Hyperparameters were tuned using 5-fold GridSearchCV over feature count and regularization strength. Best config: `max_features=15000`, `C=1.5`.

**Advanced — DistilBERT**  
Fine-tuned `distilbert-base-uncased` for sequence classification using HuggingFace Transformers and PyTorch. Trained for 3 epochs with a batch size of 16 and a learning rate of 5e-5, with mixed precision (fp16) on GPU.

## Results

| Model | F1 Score | Accuracy |
|---|---|---|
| TF-IDF + Logistic Regression | 0.775 | 0.82 |
| DistilBERT | 0.943 | 0.95 |

DistilBERT improves the F1 score by roughly 21% over the tuned baseline.

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
nltk
tensorflow
transformers
torch
datasets
joblib
```

## Usage

The notebook walks through the full pipeline: loading data, preprocessing, training both models, evaluating on a validation split, and generating predictions for submission.

To run it, open `natural-language-processing-with-disaster-tweets.ipynb` in Jupyter or on Kaggle. GPU is recommended for the DistilBERT training step (takes ~2 minutes on a T4).

## File Structure

```
.
├── natural-language-processing-with-disaster-tweets.ipynb
└── README.md
```
