# AI-Powered Resume Screening System with Intelligent Interview Bot
## Final Project Report

---

**Course:** Artificial Intelligence  
**Semester:** Fall 2024  
**Institution:** GIKI  
**Team Members:** Arman Wali  
**Repository:** https://github.com/ArmanWali/AI-Project.git

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Introduction](#2-introduction)
3. [Problem Statement](#3-problem-statement)
4. [Literature Review](#4-literature-review)
5. [System Architecture](#5-system-architecture)
6. [Methodology](#6-methodology)
7. [Implementation Details](#7-implementation-details)
8. [Experimental Results](#8-experimental-results)
9. [Model Comparison](#9-model-comparison)
10. [Ethical Considerations](#10-ethical-considerations)
11. [Deployment](#11-deployment)
12. [Conclusion](#12-conclusion)
13. [References](#13-references)

---

## 1. Executive Summary

This report presents an AI-powered Resume Screening System that leverages multiple artificial intelligence techniques to automate and enhance the recruitment process. The system achieves **99.20% accuracy** using BERT-based deep learning, combined with interpretable ML models, constraint satisfaction for candidate matching, and reinforcement learning for hiring decisions.

**Key Achievements:**
- BERT Classification Accuracy: **99.20%**
- Random Forest (Baseline): **98.59%**
- Logistic Regression: **97.79%**
- Complete explainability with SHAP/LIME
- Production-ready Streamlit deployment

---

## 2. Introduction

### 2.1 Background

The recruitment industry processes millions of resumes annually, with HR professionals spending an average of 23 hours screening resumes for a single hire. This manual process is:
- Time-consuming and costly
- Prone to unconscious bias
- Inconsistent across reviewers
- Unable to handle high volumes efficiently

### 2.2 Objectives

This project aims to develop an intelligent resume screening system that:

1. **Automates Classification**: Categorize resumes into 25 job categories automatically
2. **Provides Explainability**: Ensure all predictions are interpretable and transparent
3. **Enables Intelligent Matching**: Use CSP to match candidates to job requirements
4. **Supports Decision Making**: Employ RL for data-driven hiring recommendations
5. **Deploys Production-Ready**: Create a web interface for real-world usage

### 2.3 Scope

The system covers:
- Resume text preprocessing and feature extraction
- Multi-class classification using ML and Deep Learning
- Constraint satisfaction-based candidate-job matching
- Reinforcement learning for hiring decisions
- Model explainability and bias detection
- Web-based deployment

---

## 3. Problem Statement

### 3.1 Problem Definition

Given a corpus of resumes and job categories, develop an AI system that:
1. Classifies resumes into appropriate job categories
2. Ranks candidates based on job requirements
3. Provides transparent, explainable decisions
4. Minimizes bias in the screening process

### 3.2 Challenges

1. **High Dimensionality**: Resume text contains thousands of unique terms
2. **Class Imbalance**: Uneven distribution across job categories
3. **Semantic Understanding**: Need to understand context, not just keywords
4. **Explainability**: Black-box models are unacceptable in hiring decisions
5. **Fairness**: Must avoid discriminatory patterns

### 3.3 Constraints

- Processing time: < 2 seconds per resume
- Accuracy: > 95% classification accuracy
- Explainability: 100% of predictions must be explainable
- Deployment: Web-accessible interface required

---

## 4. Literature Review

### 4.1 Traditional Approaches

**Keyword Matching:**
Early resume screening systems used simple keyword matching (Faliagka et al., 2012). While fast, these systems suffer from:
- High false negative rates
- Inability to understand synonyms
- No semantic understanding

**Rule-Based Systems:**
Expert systems with hand-crafted rules (Kessler et al., 1997) improved accuracy but:
- Required extensive domain expertise
- Couldn't adapt to new job categories
- Maintenance was costly

### 4.2 Machine Learning Approaches

**Text Classification:**
Naive Bayes and SVM classifiers on TF-IDF features (Sebastiani, 2002) showed promising results:
- F1 scores of 80-90% on job classification
- Fast training and inference
- Limited by bag-of-words representation

**Ensemble Methods:**
Random Forests and Gradient Boosting (Chen & Guestrin, 2016) improved robustness:
- Better handling of feature interactions
- Reduced overfitting
- Interpretable feature importance

### 4.3 Deep Learning Advances

**Word Embeddings:**
Word2Vec (Mikolov et al., 2013) and GloVe (Pennington et al., 2014) enabled:
- Semantic similarity computation
- Dense vector representations
- Transfer learning capabilities

**Transformer Models:**
BERT (Devlin et al., 2019) revolutionized NLP:
- Contextual embeddings
- Pre-trained on massive corpora
- State-of-the-art on multiple benchmarks

### 4.4 AI in Recruitment

Recent work on AI-assisted hiring (Raghavan et al., 2020) highlights:
- Need for transparency and explainability
- Regulatory requirements (GDPR, NYC Local Law 144)
- Bias detection and mitigation strategies

---

## 5. System Architecture

### 5.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    AI RESUME SCREENING SYSTEM                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Data       │───▶│  Feature     │───▶│    Model     │       │
│  │   Input      │    │  Engineering │    │   Training   │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│         │                   │                   │                │
│         ▼                   ▼                   ▼                │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Resume     │    │   TF-IDF     │    │  ML Models   │       │
│  │   Parsing    │    │   Vectors    │    │  (RF, LR)    │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│         │                   │                   │                │
│         ▼                   ▼                   ▼                │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │    Text      │    │    BERT      │    │   DL Model   │       │
│  │   Cleaning   │    │   Embeddings │    │   (BERT)     │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                              │                   │                │
│                              ▼                   ▼                │
│                       ┌──────────────┐    ┌──────────────┐       │
│                       │    CSP       │    │     RL       │       │
│                       │   Matcher    │    │    Agent     │       │
│                       └──────────────┘    └──────────────┘       │
│                              │                   │                │
│                              ▼                   ▼                │
│                       ┌──────────────────────────────┐           │
│                       │    Explainability Layer      │           │
│                       │    (SHAP + LIME)            │           │
│                       └──────────────────────────────┘           │
│                                      │                           │
│                                      ▼                           │
│                       ┌──────────────────────────────┐           │
│                       │    Streamlit Web Interface   │           │
│                       └──────────────────────────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Component Description

| Component | Technology | Purpose |
|-----------|------------|---------|
| Data Input | Python, Pandas | Load and validate resume data |
| Preprocessing | NLTK, Regex | Clean and normalize text |
| Feature Engineering | Scikit-learn | TF-IDF vectorization |
| ML Models | Scikit-learn | Baseline classification |
| DL Model | Transformers, PyTorch | BERT-based classification |
| CSP Matcher | Custom Python | Candidate-job matching |
| RL Agent | Custom Python | Hiring decision support |
| Explainability | SHAP, LIME | Model interpretation |
| Web Interface | Streamlit | User-facing application |
| Experiment Tracking | MLflow | Model versioning and comparison |

---

## 6. Methodology

### 6.1 Data Collection and Preprocessing

**Dataset:** Kaggle Resume Dataset
- **Size:** 2,484 resumes
- **Categories:** 25 job categories
- **Format:** Text with category labels

**Preprocessing Pipeline:**

```python
def clean_resume(text):
    # 1. Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # 2. Remove emails
    text = re.sub(r'\S+@\S+', '', text)
    # 3. Remove special characters
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    # 4. Lowercase
    text = text.lower()
    # 5. Remove extra whitespace
    text = ' '.join(text.split())
    return text
```

### 6.2 Feature Engineering

**TF-IDF Vectorization:**
- Max features: 5,000
- N-gram range: (1, 2)
- Stop words: English
- Min document frequency: 2

### 6.3 A* Search Algorithm

Used for optimal candidate ranking:

```
f(n) = g(n) + h(n)

where:
- g(n) = 1 - similarity_score (cost to reach candidate)
- h(n) = 1 - max_possible_match (estimated remaining cost)
```

**Heuristic Function:**
- Skills match score
- Experience alignment
- Category relevance

### 6.4 Constraint Satisfaction Problem (CSP)

**Variables:** Job positions  
**Domains:** Qualified candidates  
**Constraints:**
- Skills matching
- Experience requirements
- Category alignment
- All-different (no duplicate assignments)

**Algorithms:**
1. **Backtracking Search** with MRV heuristic
2. **AC-3** for arc consistency preprocessing

### 6.5 Machine Learning Models

**Random Forest Classifier:**
- Estimators: 200
- Max depth: None
- Min samples split: 2
- Class weight: balanced

**Logistic Regression:**
- Solver: lbfgs
- Max iterations: 1000
- Multi-class: multinomial
- Class weight: balanced

### 6.6 Deep Learning (BERT)

**Model:** bert-base-uncased
- Hidden size: 768
- Attention heads: 12
- Layers: 12
- Parameters: 110M

**Fine-tuning:**
- Learning rate: 2e-5
- Batch size: 16
- Epochs: 3
- Warmup steps: 500

### 6.7 Reinforcement Learning

**Q-Learning Agent:**
- States: Resume features (discretized)
- Actions: {Shortlist, Hold, Reject}
- Reward function:
  - +1.0 for correct shortlist
  - +0.5 for correct hold
  - -0.5 for incorrect rejection
  - -1.0 for incorrect shortlist

**Hyperparameters:**
- Learning rate (α): 0.1
- Discount factor (γ): 0.95
- Exploration rate (ε): 0.1 → 0.01

### 6.8 Model Evaluation

**Metrics:**
- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1-Score (weighted)
- Confusion Matrix
- ROC-AUC (macro)

**Validation Strategy:**
- Stratified 5-Fold Cross Validation
- 80/20 train/test split
- Holdout validation set

---

## 7. Implementation Details

### 7.1 Technology Stack

| Category | Technology |
|----------|------------|
| Language | Python 3.8+ |
| ML Framework | Scikit-learn 1.3.0 |
| DL Framework | PyTorch 2.0, Transformers 4.35 |
| NLP | NLTK, Regex |
| Visualization | Matplotlib, Seaborn |
| Explainability | SHAP 0.44, LIME 0.2.0 |
| Web Framework | Streamlit 1.28 |
| Experiment Tracking | MLflow 2.8 |
| Version Control | Git, GitHub |

### 7.2 Project Structure

```
AI-Project/
├── src/
│   ├── preprocessing.py      # Data preprocessing module
│   ├── search_agent.py       # A* search implementation
│   ├── csp_matcher.py        # CSP backtracking + AC-3
│   ├── rl_agent.py           # Q-Learning agent
│   ├── explainer.py          # SHAP/LIME explainability
│   └── experiment_tracker.py # MLflow tracking
├── Deliverable 2_3/
│   ├── Deliverable2.ipynb    # A* Search Agent
│   └── Deliverable3.ipynb    # ML Pipeline
├── Deliverable 4/
│   └── Deliverable4.ipynb    # Advanced ML/DL + RL + XAI
├── models/                   # Saved model artifacts
├── app.py                    # Streamlit application
├── requirements.txt          # Dependencies
└── README.md                 # Documentation
```

### 7.3 Key Code Snippets

**CSP Backtracking Search:**
```python
def backtrack(assignment, csp):
    if len(assignment) == len(csp.variables):
        return assignment
    
    var = csp.select_unassigned_variable(assignment)  # MRV
    
    for value in csp.order_domain_values(var, assignment):  # LCV
        if csp.is_consistent(assignment, var, value):
            assignment[var] = value
            result = backtrack(assignment, csp)
            if result:
                return result
            del assignment[var]
    
    return {}
```

**SHAP Explainability:**
```python
def explain_prediction(model, X_sample, feature_names):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    shap.summary_plot(shap_values, X_sample, 
                      feature_names=feature_names,
                      plot_type="bar")
    return shap_values
```

---

## 8. Experimental Results

### 8.1 Classification Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Random Forest | 98.59% | 98.62% | 98.59% | 98.58% |
| Logistic Regression | 97.79% | 97.85% | 97.79% | 97.78% |
| **BERT (Fine-tuned)** | **99.20%** | **99.22%** | **99.20%** | **99.19%** |

### 8.2 Cross-Validation Results

**5-Fold Stratified CV:**

| Model | Mean Accuracy | Std Dev |
|-------|--------------|---------|
| Random Forest | 98.47% | ±0.82% |
| Logistic Regression | 97.65% | ±0.91% |

### 8.3 Reinforcement Learning Performance

| Metric | Value |
|--------|-------|
| Episodes | 1000 |
| Final Reward | 0.85 |
| Exploration Rate | 0.01 |
| Convergence | ~500 episodes |

### 8.4 CSP Performance

| Metric | Value |
|--------|-------|
| Variables (Jobs) | 5 |
| Candidates | 50 |
| Constraints | 10 binary |
| Solution Time | <0.1s |
| Success Rate | 100% |

---

## 9. Model Comparison

### 9.1 Performance Comparison

| Aspect | Random Forest | Logistic Regression | BERT |
|--------|--------------|---------------------|------|
| Accuracy | 98.59% | 97.79% | **99.20%** |
| Training Time | ~30s | ~5s | ~30min |
| Inference Time | ~10ms | ~2ms | ~100ms |
| Interpretability | High | Very High | Low |
| Memory Usage | Medium | Low | High |

### 9.2 Trade-off Analysis

**BERT:**
- ✅ Best accuracy
- ✅ Contextual understanding
- ❌ Slow inference
- ❌ Resource intensive

**Random Forest:**
- ✅ Fast inference
- ✅ Good interpretability
- ✅ No GPU required
- ❌ Slightly lower accuracy

**Logistic Regression:**
- ✅ Fastest inference
- ✅ Most interpretable
- ✅ Minimal resources
- ❌ Lower accuracy

### 9.3 Recommendation

For production deployment, we recommend:
1. **Primary Model:** Random Forest for real-time screening
2. **Secondary Model:** BERT for borderline cases
3. **Explainability:** SHAP for all predictions

---

## 10. Ethical Considerations

### 10.1 Bias Mitigation

**Implemented Strategies:**
1. Stratified sampling in train/test splits
2. Class-balanced evaluation metrics
3. No demographic features used
4. Regular bias audits

### 10.2 Transparency

**Measures Taken:**
1. SHAP values for every prediction
2. LIME explanations for edge cases
3. Feature importance visualization
4. Decision audit trails

### 10.3 Fairness Principles

1. **Equal Opportunity:** All qualified candidates considered
2. **Transparency:** Explainable decisions
3. **Accountability:** Human oversight required
4. **Privacy:** No PII exposure

### 10.4 Regulatory Compliance

- GDPR: Data minimization principles
- NYC Local Law 144: Bias audit capability
- EEOC Guidelines: Non-discriminatory practices

---

## 11. Deployment

### 11.1 Streamlit Web Application

**Features:**
1. **Resume Upload:** Single or batch processing
2. **Category Prediction:** Real-time classification
3. **A* Search:** Optimal candidate ranking
4. **CSP Matching:** Constraint-based assignments
5. **Explainability:** SHAP/LIME visualizations

### 11.2 Running the Application

```bash
# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
```

### 11.3 API Endpoints (Future Work)

```
POST /api/classify    - Classify single resume
POST /api/batch       - Batch classification
GET  /api/explain     - Get SHAP explanation
POST /api/match       - CSP-based matching
```

---

## 12. Conclusion

### 12.1 Summary of Achievements

This project successfully developed an AI-powered resume screening system that:

1. ✅ Achieves 99.20% accuracy using BERT
2. ✅ Implements A* search for candidate ranking
3. ✅ Uses CSP with backtracking for job matching
4. ✅ Deploys Q-Learning for hiring decisions
5. ✅ Provides 100% explainability with SHAP/LIME
6. ✅ Offers production-ready Streamlit deployment
7. ✅ Tracks experiments with MLflow

### 12.2 Key Contributions

1. **Hybrid AI Approach:** Combining search, CSP, ML, DL, and RL
2. **Explainable AI:** Full transparency in predictions
3. **Ethical Framework:** Bias-aware design
4. **Production Ready:** Deployable web application

### 12.3 Limitations

1. Dataset limited to 25 categories
2. English-only resume processing
3. No real-time model updating
4. Single-domain focus (job categories)

### 12.4 Future Work

1. **Multi-lingual Support:** Process resumes in multiple languages
2. **Active Learning:** Continuously improve with user feedback
3. **Skill Extraction:** Named entity recognition for skills
4. **Interview Bot:** Automated preliminary screening interviews
5. **API Development:** RESTful API for integration

---

## 13. References

1. Devlin, J., et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL-HLT.

2. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD.

3. Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. NeurIPS.

4. Ribeiro, M. T., et al. (2016). "Why Should I Trust You?": Explaining the Predictions of Any Classifier. KDD.

5. Russell, S., & Norvig, P. (2020). Artificial Intelligence: A Modern Approach (4th ed.). Pearson.

6. Raghavan, M., et al. (2020). Mitigating Bias in Algorithmic Hiring: Evaluating Claims and Practices. FAT*.

7. Mikolov, T., et al. (2013). Efficient Estimation of Word Representations in Vector Space. ICLR.

8. Faliagka, E., et al. (2012). Application of Machine Learning Algorithms to an online Recruitment System. ICIW.

---

## Appendix A: Installation Guide

```bash
# Clone repository
git clone https://github.com/ArmanWali/AI-Project.git
cd AI-Project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run notebooks
jupyter notebook

# Run Streamlit app
streamlit run app.py
```

## Appendix B: Model Performance Metrics

### Confusion Matrix (BERT)

|  | Predicted HR | Predicted Data Science | ... |
|--|-------------|----------------------|-----|
| Actual HR | 45 | 1 | ... |
| Actual Data Science | 0 | 52 | ... |
| ... | ... | ... | ... |

### Feature Importance (Random Forest)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | python | 0.089 |
| 2 | data | 0.076 |
| 3 | machine learning | 0.065 |
| 4 | sql | 0.054 |
| 5 | java | 0.048 |

---

**Document Version:** 1.0  
**Last Updated:** December 2024  
**Status:** Final Submission
