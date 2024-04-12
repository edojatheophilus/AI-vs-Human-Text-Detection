# AI-vs-Human-Text-Detection
In the realm of Natural Language Processing (NLP) and Machine Learning (ML), the ability to differentiate between human-written and AI-generated text is pivotal.
The objective of this project is to develop robust methodologies for detecting AI-generated text within a corpus of textual data. 
By leveraging data analysis techniques and machine learning models, we aim to accurately classify text entries as either human-written or AI-generated.
The ability to discern between human and AI-generated text holds implications across various domains, including journalism, social media,  academia, legal compliance, marketing & advertising and  entertainment .

The dataset utilized in this analysis comprises a substantial collection of texts sourced from Kaggle, encompassing both AI-generated and human-written content.

![image](https://github.com/edojatheophilus/AI-vs-Human-Text-Detection/assets/139919035/a25f0541-3e9f-42f5-87e9-a23ebfaa5a69)

**Dataset Source:** https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text

Highlights the significance of discerning between human and AI-generated text in the context of NLP and ML advancements. 

The initial exploration steps including data loading, basic statistics, and visualization of text distribution.

**Data Preprocessing**

Outlines data cleaning and preprocessing steps, including punctuation and linking word count comparison, text length distribution, POS tag frequency, named entity recognition, and parsing techniques.

**Feature Engineering:**

Details the extraction of features such as character count, word count, average word length, TF-IDF vectorization, dimensionality reduction using PCA, and normalization of data.

**Modeling & Evaluation:**

Evaluates various machine learning models including 
* Random Forest
* SVM
* RNN
  
Provides performance metrics and interpretation methods like feature importance analysis, SHAP values, LIME analysis, decision tree visualization, and confusion matrix comparison.

**Model Deployment:**

Setup using Python's Flask framework, incorporating the Random Forest Classifier, TF-IDF Vectorizer, and PCA.




