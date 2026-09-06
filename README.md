# 🛍️ Product Review Sentiment Analysis using Transformers

A case study project from **Generative AI for Business Applications — Week 3: Embeddings to Transformers**, which builds an end-to-end sentiment classification system for e-commerce product reviews using pre-trained sentence embeddings and classical machine learning models.

---

## 📌 Project Description

In digital commerce, customer reviews are one of the richest sources of feedback a business has — but reading through thousands of reviews manually to gauge sentiment isn't scalable. This project builds an automated pipeline that reads raw, unstructured review text and classifies it as **Positive**, **Neutral**, or **Negative**.

Instead of traditional text-vectorization methods like TF-IDF or Bag-of-Words, the pipeline uses a **pre-trained Sentence Transformer** (`all-MiniLM-L6-v2`) to convert each review into a dense 384-dimensional embedding that captures its semantic meaning. These embeddings are then used as input features to train and compare two classical ML classifiers — **Random Forest** and **Gradient Boosting** — and the better-generalizing model is selected for deployment.

---

## 🎯 Objective

1. **Generate meaningful text representations** of customer reviews using `all-MiniLM-L6-v2` sentence embeddings.
2. **Train and compare multiple classifiers** — Random Forest and Gradient Boosting — on these embeddings.
3. **Evaluate performance** using Accuracy and F1 score (weighted) on both train and test sets, checking for overfitting.
4. **Select the best-performing model** for deployment based on test-set performance and train/test consistency.

---

## 🗂️ Dataset

**File:** `Product_Reviews.csv`

| Column | Description |
|---|---|
| `Product ID` | Unique identifier for each product |
| `Product Review` | Free-form customer review text |
| `Sentiment` | Label — Positive / Neutral / Negative |

**Key facts:**
- 1,007 rows × 3 columns originally → **1,005 unique rows** after removing 2 duplicates
- No missing values in any column
- Class distribution is **imbalanced**: ~850 Positive vs. ~75 Neutral and ~75 Negative reviews
- Because of this imbalance, **F1 score** is tracked alongside accuracy for a fair evaluation

---

## 🏗️ Project Workflow

```
Raw Reviews (CSV)
      │
      ▼
Data Cleaning (duplicates, nulls, index reset)
      │
      ▼
Exploratory Data Analysis (class distribution)
      │
      ▼
Sentence Embeddings — all-MiniLM-L6-v2 (384-dim vectors)
      │
      ▼
Stratified Train/Test Split (80/20)
      │
      ▼
Model Training — Random Forest & Gradient Boosting
      │
      ▼
Evaluation — Accuracy & F1 Score (train vs. test)
      │
      ▼
Final Model Selection
```

---

## ⚙️ Tech Stack

| Purpose | Library |
|---|---|
| Data handling | `pandas`, `numpy` |
| Text embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Modeling | `scikit-learn` (`RandomForestClassifier`, `GradientBoostingClassifier`) |
| Evaluation | `accuracy_score`, `f1_score`, `classification_report` |
| Visualization | `matplotlib`, `seaborn` |
| Environment | Google Colab (Google Drive for data access) |

---

## 📊 Results

| Model | Train Accuracy | Train F1 | Test Accuracy | Test F1 | Generalization Drop (Acc / F1) |
|---|---|---|---|---|---|
| **Random Forest + Transformer** | 100% | 100% | 86.6% | 81.8% | 13.4% / 18.2% |
| **Gradient Boost + Transformer** | 99.88% | 99.87% | 84.1% | 80.3% | 15.8% / 19.6% |

**✅ Final choice: Random Forest** — it shows a lower generalization gap between training and test performance, making it the more consistent and deployable model of the two.

---

## 🔑 Key Takeaways

- Sentence embeddings capture semantic meaning far better than word-frequency-based methods, without needing manual feature engineering.
- Class imbalance (Positive-heavy) means accuracy alone can be misleading — F1 score is essential for judging performance on minority classes (Neutral/Negative), which are often the most business-critical to catch.
- Both models overfit somewhat on the training data (100% / ~99.9% train scores) but Random Forest generalizes better to unseen reviews.
- Future improvements could include class weighting, macro-F1 tracking per class, XGBoost, SVMs, or fine-tuning a transformer end-to-end instead of using it purely as a feature extractor.

---

## 🚀 How to Run

1. Open the notebook in **Google Colab** (recommended) or Jupyter.
2. Mount Google Drive and place `Product_Reviews.csv` at:
   `/content/drive/MyDrive/Dataset/GenAIDataset/Product_Reviews.csv`
   *(or update the file path in the "Loading the dataset" cell to match your own location)*
3. Run the first cell to install `sentence-transformers`, then **restart the runtime**.
4. Run all remaining cells sequentially from top to bottom.

**Requirements** (auto-installed in the notebook):
```
pandas
numpy
scikit-learn
sentence-transformers
matplotlib
seaborn
```

---

## 📁 Files

| File | Description |
|---|---|
| `Product_Review_Sentiment_Analysis_Transformers.ipynb` | Full notebook — EDA, embedding generation, model training & evaluation |
| `Product_Reviews.csv` | Raw dataset (not included — supply your own copy) |
| `README.md` | This file |

---

## ✍️ Author's Note

This project was built as part of a Generative AI for Business Applications case study, demonstrating how transformer-based embeddings can be combined with classical ML for a lightweight, interpretable, and effective sentiment analysis pipeline — without needing to fine-tune a full deep learning model.
