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

## Notebooks

- **01_eda.ipynb** – corpus exploration, class balance, text length distributions, vocabulary analysis
- **02_preprocessing.ipynb** – text cleaning, normalisation, train/validation/test split strategy
- **03_feature_engineering.ipynb** – BoW, TF-IDF, and word embedding representations
- **04_classification_models_&_evaluation.ipynb** – traditional ML classifiers, cross-validation, and evaluation
- **05_deep_architectures.ipynb** – stacked RNN and BiLSTM architectures with Keras
- **06_transformer_embeddings.ipynb** – frozen transformer encoders as feature extractors
- **07_transformer_finetuning.ipynb** – end-to-end fine-tuning of DeBERTa-v3, FinBERT, and Twitter-RoBERTa
- **08_decoder_classification.ipynb** – zero/few-shot classification with decoder LLMs *(extra credit)*
- **09_agentic_classifier.ipynb** – agentic AI pipeline for classification *(extra credit)*

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
