# Machine-Learning-and-NLP-Project
This project consists of two machine learning and artificial intelligence components developed using Python. The project demonstrates the application of machine learning, natural language processing (NLP), sentiment analysis, text embeddings, and Retrieval-Augmented Generation (RAG).
The project is divided into:

* **Component A:** South African Road Accident Severity Prediction
* **Component B:** Hansard Sentiment Analysis and Retrieval-Augmented Generation

Both components use real-world data and demonstrate different applications of machine learning and AI.

---

# Component A: Road Accident Severity Prediction

## Description

Component A focuses on predicting road accident severity using the **South Africa Road Accidents Dataset - 2017**.

The dataset contains information about road accidents, including factors such as location, police force, number of vehicles, vehicle type, speed, speed zone, casualties, and year.

Three machine learning classification algorithms were implemented and evaluated:

* Random Forest
* XGBoost
* CatBoost

## Methodology

The following steps were performed:

1. Loaded the South African road accident dataset.
2. Removed duplicate records.
3. Identified the accident severity target variable.
4. Separated the features from the target.
5. Handled missing and infinite values.
6. Converted date/time features into numerical values.
7. Encoded categorical variables using `OrdinalEncoder`.
8. Split the data into training and testing sets.
9. Trained Random Forest, XGBoost, and CatBoost classification models.
10. Evaluated the models using:

* Accuracy
* Precision
* Recall
* F1 Score

11. Generated confusion matrices.
12. Analysed XGBoost feature importance.
13. Compared training and testing performance.

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost
* CatBoost
* Jupyter Notebook

---

# Component B: Hansard NLP and RAG System

## Description

Component B applies Natural Language Processing and transformer-based models to South African parliamentary Hansard transcripts.

Hansard transcripts are downloaded, cleaned, divided into sentences, and processed for sentiment analysis.

The component includes:

* Text preprocessing
* Rule-based sentiment labelling
* DistilBERT sentiment classification
* Sentence embeddings
* FAISS similarity search
* Retrieval-Augmented Generation (RAG)
* FLAN-T5 text generation

## Methodology

The following steps were performed:

1. Downloaded Hansard transcripts.
2. Extracted text from the documents.
3. Cleaned and normalised the text.
4. Split the documents into individual sentences.
5. Created sentiment labels using positive and negative keyword rules.
6. Divided the dataset into training and testing sets.
7. Tokenised the text using the DistilBERT tokenizer.
8. Fine-tuned a DistilBERT classification model.
9. Evaluated the sentiment model using:

   * Accuracy
   * Precision
   * Recall
   * F1 Score
10. Generated sentence embeddings using `all-MiniLM-L6-v2`.
11. Created a FAISS vector index for similarity-based retrieval.
12. Used FLAN-T5 to generate answers based on retrieved Hansard passages.
13. Tested the RAG system using questions about topics such as:

* Education
* Unemployment
* Healthcare
* Economic growth
* Poverty

## RAG Architecture

The RAG system follows this general process:

```text
User Question
      ↓
Sentence Embedding
      ↓
FAISS Similarity Search
      ↓
Relevant Hansard Passages
      ↓
Context + Question
      ↓
FLAN-T5
      ↓
Generated Answer
```

---

# Models Used

| Model            | Purpose                               |
| ---------------- | ------------------------------------- |
| Random Forest    | Road accident severity classification |
| XGBoost          | Road accident severity classification |
| CatBoost         | Road accident severity classification |
| DistilBERT       | Hansard sentiment classification      |
| all-MiniLM-L6-v2 | Sentence embeddings                   |
| FAISS            | Similarity-based document retrieval   |
| FLAN-T5          | RAG answer generation                 |

---

# Evaluation

## Component A

The road accident classification models are evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion matrices
* Feature importance
* Training vs testing accuracy

## Component B

The sentiment classification model is evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion matrix
* Classification report

The RAG system is evaluated by examining retrieved Hansard passages and checking whether relevant topic keywords are present in the retrieved results.

---

# Project Structure

```text
ML700/
│
├── ML700.ipynb
│
├── data/
│   ├── raw_hansard.parquet
│   ├── processed_hansard.parquet
│   ├── component_A_model_results.csv
│   ├── component_A_feature_importance.csv
│   ├── component_A_train_test_results.csv
│   ├── component_B_sentiment_results.csv
│   ├── component_B_rag_results.csv
│   └── component_B_retrieval_results.csv
│
├── models/
│   ├── sentiment_model/
│   └── hansard_faiss.index
│
└── README.md
```

---

# Installation

Install the required Python libraries using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost catboost
pip install requests beautifulsoup4 nltk torch
pip install transformers datasets sentence-transformers faiss-cpu
pip install pyarrow openpyxl tqdm
```

---

# Running the Project

1. Install Python and Jupyter Notebook.
2. Install the required libraries.
3. Place the South African road accident dataset in the appropriate location.
4. Open `ML700.ipynb` in Jupyter Notebook.
5. Run the notebook cells in order.
6. Component A will train and evaluate the road accident classification models.
7. Component B will download and process the Hansard transcripts, train the sentiment model, create the FAISS index, and run the RAG system.
8. The evaluation results and generated CSV files will be stored in the `data` directory.

---

# Important Note

The sentiment labels in Component B are generated using predefined positive and negative keyword rules. Therefore, the DistilBERT evaluation measures how well the model learns these automatically generated labels rather than independently human-validated sentiment annotations.

The RAG retrieval evaluation uses a simple keyword-based coverage check and should therefore be interpreted as a basic retrieval evaluation rather than a formal measure of retrieval accuracy.

---

# Learning Outcomes

This project demonstrates practical experience with:

* Data preprocessing
* Exploratory data analysis
* Classification
* Ensemble machine learning
* Model evaluation
* Feature importance
* Natural Language Processing
* Transformer models
* Sentiment analysis
* Text embeddings
* Vector databases/search
* Retrieval-Augmented Generation
* Python machine learning workflows

---

# Author

**Brighton Jonathan**

BSc Information Technology Student
Richfield Graduate Institute of Technology
Johannesburg, South Africa
