# AI Resume Screening System - Deliverables 2 & 3

## 📁 Project Overview

This project implements an AI-powered resume screening system that uses intelligent agents, search algorithms, and machine learning to automate candidate evaluation.

---

## ✅ Deliverable 2: Agent & Search Algorithm

### What's Implemented:

**File:** `Deliverable2.ipynb`

1. **Intelligent Search Agent**
   - Type: Goal-based agent
   - Goal: Find best matching candidates for job requirements
   - Algorithm: A* Search

2. **A* Search Algorithm Components:**
   - **Heuristic Function:** 
     ```
     h(n) = 0.6 × skill_match + 0.4 × experience_match
     ```
   - **Admissibility:** Never overestimates match quality
   - **Time Complexity:** O(N log N)
   - **Space Complexity:** O(N)

3. **Features:**
   - Resume parsing (extract skills and experience)
   - Candidate class for organizing data
   - Search agent that ranks candidates
   - Performance visualization

### How It Works:

1. Loads resume dataset from Kaggle
2. Extracts skills and years of experience from each resume
3. Creates Candidate objects
4. Defines job requirements (skills + min experience)
5. Runs A* search to find top 10 matching candidates
6. Displays results with match scores

### Running the Code:

```python
# Just run all cells in order in Deliverable2.ipynb
# The notebook will:
# 1. Download the dataset
# 2. Parse resumes
# 3. Run A* search
# 4. Show top 10 candidates
```

---

## ✅ Deliverable 3: Data Preprocessing + ML Models

### What's Implemented:

**File:** `Deliverable3.ipynb`

1. **Data Preprocessing Pipeline:**
   - Text cleaning (remove URLs, special characters)
   - TF-IDF vectorization (300 features)
   - Structured feature extraction (skills count, length, word count)
   - Train-test split (80/20)

2. **Baseline Model 1: Random Forest**
   - 100 decision trees
   - Max depth: 20
   - Handles multi-class classification

3. **Baseline Model 2: Logistic Regression**
   - Linear classification model
   - Max iterations: 1000
   - Multi-class support

4. **Evaluation Metrics:**
   - Accuracy
   - Precision
   - Recall
   - F1-Score
   - Confusion Matrix

### Pipeline Steps:

```
Resume Text 
    ↓
Text Cleaning
    ↓
Feature Extraction (TF-IDF + Structured)
    ↓
Train-Test Split
    ↓
Train Models (RF + LR)
    ↓
Evaluate Performance
    ↓
Save Models
```

### Key Results:

- **Dataset:** 2,484 resumes across 25 job categories
- **Features:** 303 total (300 TF-IDF + 3 structured)
- **Best Model:** Random Forest (typically 95%+ accuracy)
- **Models Saved:** Ready for future use

### Running the Code:

```python
# Just run all cells in order in Deliverable3.ipynb
# The notebook will:
# 1. Load and clean data
# 2. Extract features
# 3. Train both models
# 4. Compare performance
# 5. Save models to disk
```

---

## 📊 Performance Metrics Explained

### Accuracy
- **What it means:** Percentage of correct predictions
- **Formula:** (Correct Predictions) / (Total Predictions)

### Precision
- **What it means:** Of candidates we recommended, how many were actually suitable
- **Formula:** True Positives / (True Positives + False Positives)

### Recall
- **What it means:** Of all suitable candidates, how many did we find
- **Formula:** True Positives / (True Positives + False Negatives)

### F1-Score
- **What it means:** Balance between Precision and Recall
- **Formula:** 2 × (Precision × Recall) / (Precision + Recall)
- **Why important:** Best overall metric for imbalanced datasets

---

## 📦 Required Libraries

All libraries are automatically installed when you run the notebooks:

```
- pandas: Data manipulation
- numpy: Numerical operations
- kagglehub: Dataset download
- scikit-learn: Machine learning
- matplotlib: Visualization
- seaborn: Advanced visualization
- scipy: Sparse matrices
- joblib: Model saving
```

---

## 🎯 What Each Component Does

### Deliverable 2 Components:

| Component | Purpose | Code Location |
|-----------|---------|---------------|
| `extract_skills()` | Find skills in resume text | Cell 3 |
| `extract_experience_years()` | Find years of experience | Cell 3 |
| `Candidate` class | Organize candidate data | Cell 4 |
| `SearchAgent` class | A* search implementation | Cell 5 |
| `heuristic()` | Calculate match score | Inside SearchAgent |
| `a_star_search()` | Main search algorithm | Inside SearchAgent |

### Deliverable 3 Components:

| Component | Purpose | Code Location |
|-----------|---------|---------------|
| `clean_resume_text()` | Remove noise from text | Step 2 |
| `TfidfVectorizer` | Convert text to numbers | Step 3 |
| `count_skills()` | Count technical skills | Step 3 |
| `RandomForestClassifier` | ML Model 1 | Step 5 |
| `LogisticRegression` | ML Model 2 | Step 6 |
| Confusion Matrix | Visualize predictions | Step 8 |
| Feature Importance | Show important features | Step 9 |

---

## 🚀 Next Steps (Future Deliverables)

### Deliverable 4 (Due Nov 20):
- [ ] Implement CSP for interview scheduling
- [ ] Integrate BERT for semantic matching
- [ ] Add Reinforcement Learning agent
- [ ] Implement SHAP/LIME explainability

### Final Deliverable (Due Dec 11):
- [ ] Complete system integration
- [ ] Web interface (Flask)
- [ ] Final report
- [ ] Presentation slides
- [ ] Demo video

---

## 💡 Understanding the Code

### Simple Explanation:

**Deliverable 2:**
Think of it like a smart recruiter that:
1. Reads all resumes
2. Compares them to job requirements
3. Finds the best matches using a clever algorithm (A*)
4. Ranks candidates by how well they fit

**Deliverable 3:**
Think of it like teaching a computer to classify resumes:
1. Clean up the resume text
2. Convert words into numbers (TF-IDF)
3. Train two different "brains" (Random Forest & Logistic Regression)
4. Test how well they can categorize resumes
5. Keep the best one for future use

---

## 📝 Report Writing Tips

### For Deliverable 2 Report:

1. **Introduction (0.5 page)**
   - Problem: Too many resumes to review manually
   - Solution: A* search agent

2. **Agent Design (1 page)**
   - Agent type: Goal-based
   - Components: Percepts, actions, goal
   - Architecture diagram

3. **A* Algorithm (1 page)**
   - Heuristic function explanation
   - Why it's admissible
   - Time/space complexity

4. **Results (1 page)**
   - Show top 10 candidates output
   - Search time
   - Performance graphs

### For Deliverable 3 Report:

1. **Dataset (0.5 page)**
   - Source: Kaggle
   - Size: 2,484 resumes
   - Categories: 25 job types

2. **Preprocessing (1 page)**
   - Text cleaning steps
   - Feature extraction methods
   - TF-IDF explanation

3. **Models (1.5 pages)**
   - Random Forest description
   - Logistic Regression description
   - Hyperparameters chosen

4. **Results (1 page)**
   - Comparison table
   - Confusion matrix
   - Best model selection

---

## ❓ FAQ

**Q: What if the dataset download fails?**
A: The kagglehub library will automatically download it. If issues persist, download manually from Kaggle and place in the project folder.

**Q: How long does training take?**
A: Random Forest: ~5-10 seconds, Logistic Regression: ~2-3 seconds

**Q: Can I use different job requirements?**
A: Yes! Just modify the `job_requirements` dictionary with your own skills and experience.

**Q: What if I want to test with more candidates?**
A: Change `df.head(500)` to `df` to use all 2,484 resumes.

---

## 📧 Submission Checklist

### Deliverable 2:
- [ ] `Deliverable2.ipynb` with all cells executed
- [ ] Screenshots of output showing top 10 candidates
- [ ] Report PDF (3-4 pages)

### Deliverable 3:
- [ ] `Deliverable3.ipynb` with all cells executed
- [ ] `models/` folder with saved models
- [ ] Screenshots of visualizations
- [ ] Report PDF (3-4 pages)

---

## 🎓 Learning Outcomes Achieved

✅ **Formulated AI problem** with measurable goals (match score)  
✅ **Implemented search algorithm** efficiently (A* with O(N log N))  
✅ **Handled real-world dataset** (2,484 resumes, cleaning, preprocessing)  
✅ **Developed ML models** (Random Forest, Logistic Regression)  
✅ **Evaluated models** using proper metrics (Accuracy, F1, etc.)  
✅ **Documented solution** with clear explanations  

---

**Good luck with your submission! 🚀**
