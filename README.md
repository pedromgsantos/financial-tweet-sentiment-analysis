# Financial Tweet Sentiment Classification

Project for the Text Mining course, Master's in Data Science and Advanced Analytics, NOVA IMS.

---

## Authors

Pedro Santos -- 20250399 -- [20250399@novaims.unl.pt](mailto:20250399@novaims.unl.pt)  
Tiago Duarte -- 20250360 -- [20250360@novaims.unl.pt](mailto:20250360@novaims.unl.pt)

---

## Overview

Three-class sentiment classification of financial tweets: Bearish (0), Bullish (1), and Neutral (2). The pipeline progresses from lexical baselines through classical ML, recurrent architectures, frozen transformer embeddings, and end-to-end fine-tuned encoders. Two extra components are included: zero/few-shot classification with a decoder LLM (Phi-3-mini) and a LangChain ReAct agentic workflow.

---

## Dataset

Training corpus from [`zeroshot/twitter-financial-news-sentiment`](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment).

| Split | Samples |
|-------|---------|
| Train | 9,543   |
| Test  | 2,388   |

| Class       | Count | Share  |
|-------------|-------|--------|
| Bearish (0) | 1,442 | 15.1%  |
| Bullish (1) | 1,923 | 20.2%  |
| Neutral (2) | 6,178 | 64.7%  |

The corpus is heavily skewed toward Neutral. F1-Macro is used as the primary metric throughout, as raw accuracy is misleading under this imbalance (a majority-class predictor already reaches 64.7%).

---

## Results

All models evaluated on a stratified 20% held-out validation split (1,909 samples). F1-Macro is the primary metric.

### Baselines and classical ML

| Model | Accuracy | Precision | Recall | F1-Macro |
|-------|----------|-----------|--------|----------|
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

### Frozen transformer embeddings + downstream classifier

| Encoder | Classifier | Accuracy | Precision | Recall | F1-Macro |
|---------|------------|----------|-----------|--------|----------|
| DistilBERT | LinearSVC | 0.791 | 0.717 | 0.722 | 0.719 |
| FinBERT | XGBoost | 0.834 | 0.789 | 0.767 | 0.777 |
| Twitter-RoBERTa | XGBoost | 0.846 | 0.803 | 0.790 | 0.796 |

### Fine-tuned encoders

| Model | Accuracy | Precision | Recall | F1-Macro |
|-------|----------|-----------|--------|----------|
| FinBERT | 0.863 | 0.819 | 0.824 | 0.820 |
| Twitter-RoBERTa | 0.885 | 0.846 | 0.864 | 0.854 |
| **DeBERTa-v3** | **0.896** | **0.867** | **0.876** | **0.871** |

### Decoder LLM -- extra credit

| Model | Accuracy | Precision | Recall | F1-Macro |
|-------|----------|-----------|--------|----------|
| Phi-3-mini zero-shot | 0.730 | 0.695 | 0.805 | 0.719 |
| Phi-3-mini few-shot | 0.837 | 0.802 | 0.795 | 0.798 |

**Final model:** DeBERTa-v3 fine-tuned on the full 9,543-tweet corpus. Test set F1-Macro: **0.8798**.

---

## Pipeline

| # | Notebook | Description |
|---|----------|-------------|
| 1 | `01_eda.ipynb` | Class distribution, tweet length, vocabulary analysis, word clouds |
| 2 | `02_preprocessing.ipynb` | Text cleaning, sentinel token substitution, train/validation split |
| 3 | `03_feature_engineering.ipynb` | BoW, TF-IDF, Word2Vec (CBOW/Skip-Gram), GloVe |
| 4 | `04_classification_models.ipynb` | Classical ML classifiers and evaluation |
| 5 | `05_deep_architectures.ipynb` | Stacked RNN and BiLSTM over GloVe sequences |
| 6 | `06_transformer_embeddings.ipynb` | Frozen encoder CLS embeddings as features |
| 7 | `07_transformer_finetuning.ipynb` | End-to-end fine-tuning of DeBERTa-v3, FinBERT, Twitter-RoBERTa |
| 8 | `08_decoder_classification.ipynb` | Zero/few-shot classification with Phi-3-mini -- extra credit |
| 9 | `09_agentic_classifier.ipynb` | LangChain ReAct agentic workflow -- extra credit |

The consolidated experiment notebook (`tm_tests_24.ipynb`) and the final pipeline (`tm_final_24.ipynb`) are in `group_24/`.

---

## Preprocessing

The `clean_text()` pipeline applies ten steps in order: encoding repair (ftfy), emoji removal, URL and mention removal, sentinel token substitution (cashtags, hashtags, percentages, monetary values, quarterly references, major indices), contraction expansion, lowercasing, punctuation removal, tokenisation, stop word removal, and lemmatisation/stemming. Negation words and market-movement terms are excluded from the stop word list.

---

## Requirements

```bash
pip install -r requirements.txt
```

| Group | Packages |
|-------|---------|
| Core | `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`, `plotly`, `tqdm` |
| NLP | `nltk`, `gensim`, `contractions`, `ftfy`, `emoji`, `langdetect`, `wordcloud` |
| ML | `scikit-learn`, `xgboost`, `lightgbm` |
| Deep learning | `tensorflow`, `torch`, `transformers`, `accelerate`, `bitsandbytes`, `sentencepiece` |
| Agents | `langchain-core`, `langchain-community`, `langchain-openai` |

Notebooks 07 and `tm_final_24` require a GPU environment (tested on Google Colab T4). `bitsandbytes` and `accelerate` are only needed for those notebooks.

---

## Submission (`group_24/`)

| File | Description |
|------|-------------|
| `tm_final_24.ipynb` | Ready-to-run pipeline: fine-tunes DeBERTa-v3 on the full training corpus and produces predictions |
| `tm_tests_24.ipynb` | Full experiment notebook covering all pipeline stages |
| `pred_24.csv` | Test set predictions (2,388 rows, columns: `id`, `y_pred`) |
| `report_24.pdf` | Written report (15 pages) |
