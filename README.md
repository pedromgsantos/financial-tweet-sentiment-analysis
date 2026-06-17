# Market Sentiment Analysis from Tweets

Project for the Text Mining course, Master's in Data Science and Advanced Analytics, NOVA IMS.

---

## Project Authors

Pedro Santos – 20250399 – [20250399@novaims.unl.pt](mailto:20250399@novaims.unl.pt)  
Tiago Duarte – 20250360 – [20250360@novaims.unl.pt](mailto:20250360@novaims.unl.pt)

---

## Project Overview

The dataset consists of financial tweets labelled with market sentiment (bearish, bullish, neutral).  
The objective is to build a text classification pipeline that accurately predicts tweet sentiment, progressively moving from traditional machine learning baselines to state-of-the-art transformer fine-tuning.

---

## Project Goals

1. **Exploratory Data Analysis** – Understand the corpus, class distribution, and linguistic characteristics of financial tweets.
2. **Preprocessing** – Clean and normalise text; produce reproducible train/validation/test splits.
3. **Feature Engineering** – Evaluate text representation techniques (BoW, TF-IDF, word embeddings).
4. **Traditional ML Classification** – Establish baselines with classical classifiers and systematic evaluation.
5. **Deep Architectures** – Experiment with stacked RNN and LSTM networks over learned embeddings.
6. **Transformer Embeddings** – Use frozen pre-trained encoder representations as features for downstream classifiers.
7. **Transformer Fine-Tuning** – Fine-tune encoder models (DeBERTa-v3, FinBERT, Twitter-RoBERTa) end-to-end.
8. **Extensions** – Decoder LLM zero/few-shot classification and an agentic AI classification pipeline (extra credit).

---

## Outcome

Fine-tuned **DeBERTa-v3** was selected as the final model, outperforming all baselines and alternative architectures across the validation set. Final predictions are in `group_24/pred_24.csv`.

---

## Results

All models evaluated on a stratified 20% held-out validation split (1,909 samples). F1-Macro is the primary metric.

**Baselines and classical ML**

| Model | Accuracy | Precision | Recall | F1-Macro |
|---|---|---|---|---|
| VADER (lexicon baseline) | 0.483 | -- | -- | 0.438 |
| KNN (k=15) | 0.798 | 0.787 | 0.647 | 0.688 |
| BiLSTM (3L stacked, GloVe) | 0.713 | 0.634 | 0.684 | 0.651 |
| Naive Bayes | 0.792 | 0.792 | 0.713 | 0.718 |
| LightGBM | 0.791 | 0.718 | 0.731 | 0.724 |
| Random Forest | 0.818 | 0.811 | 0.692 | 0.734 |
| XGBoost | 0.811 | 0.743 | 0.752 | 0.748 |
| MLP (512→256) | 0.821 | 0.783 | 0.725 | 0.750 |
| Logistic Regression (L2) | 0.810 | 0.745 | 0.758 | 0.751 |
| LinearSVC (C=0.1) | 0.822 | 0.773 | 0.749 | 0.760 |

**Frozen transformer embeddings + downstream classifier**

| Encoder | Classifier | Accuracy | Precision | Recall | F1-Macro |
|---|---|---|---|---|---|
| DistilBERT | LinearSVC | 0.791 | 0.717 | 0.722 | 0.719 |
| FinBERT | XGBoost | 0.834 | 0.789 | 0.767 | 0.777 |
| Twitter-RoBERTa | XGBoost | 0.846 | 0.803 | 0.790 | 0.796 |

**Fine-tuned encoders**

| Model | Accuracy | Precision | Recall | F1-Macro |
|---|---|---|---|---|
| FinBERT | 0.863 | 0.819 | 0.824 | 0.820 |
| Twitter-RoBERTa | 0.885 | 0.846 | 0.864 | 0.854 |
| **DeBERTa-v3** | **0.896** | **0.867** | **0.876** | **0.871** |

**Decoder LLM**

| Model | Accuracy | Precision | Recall | F1-Macro |
|---|---|---|---|---|
| Phi-3-mini zero-shot | 0.730 | 0.695 | 0.805 | 0.719 |
| Phi-3-mini few-shot | 0.837 | 0.802 | 0.795 | 0.798 |

---

## Notebooks

- **01_eda.ipynb** – corpus exploration, class balance, text length distributions, vocabulary analysis
- **02_preprocessing.ipynb** – text cleaning, normalisation, train/validation/test split strategy
- **03_feature_engineering.ipynb** – BoW, TF-IDF, and word embedding representations
- **04_classification_models_&_evaluation.ipynb** – traditional ML classifiers, cross-validation, and evaluation
- **05_deep_architectures.ipynb** – stacked RNN and BiLSTM architectures with Keras
- **06_transformer_embeddings.ipynb** – frozen transformer encoders as feature extractors
- **07_transformer_finetuning.ipynb** – end-to-end fine-tuning of DeBERTa-v3, FinBERT, and Twitter-RoBERTa
- **08_decoder_classification.ipynb** – zero/few-shot classification with decoder LLMs
- **09_agentic_classifier.ipynb** – agentic AI pipeline for classification

---

## Requirements

Install all dependencies with:

```bash
pip install -r requirements.txt
```

| Group | Packages |
|---|---|
| Core data science | `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`, `plotly`, `tqdm` |
| NLP / text processing | `nltk`, `gensim`, `contractions`, `ftfy`, `emoji`, `langdetect`, `wordcloud` |
| Machine learning | `scikit-learn`, `xgboost`, `lightgbm` |
| Deep learning | `tensorflow`, `torch`, `transformers`, `accelerate`, `bitsandbytes`, `sentencepiece` |
| LangChain / LLM agents | `langchain-core`, `langchain-classic`, `langchain-openai` |

> Notebooks 07 and `tm_final_24` were developed on GPU-enabled environments (Google Colab / Lightning AI). `bitsandbytes` and `accelerate` are only required for those notebooks.

---

## Submission (`group_24/`)

- **tm_final_24.ipynb** – clean, ready-to-run pipeline that fine-tunes DeBERTa-v3 on the full training corpus and produces predictions
- **tm_tests_24.ipynb** – experiment notebook documenting model selection and hyperparameter decisions
- **pred_24.csv** – final test set predictions
- **report_24.pdf** – written report
