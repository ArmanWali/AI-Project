# 🚀 BehterCV - User Guide

## Welcome to BehterCV!

BehterCV is your AI-powered Resume Intelligence Platform with a professional, modern interface.

---

## 📦 Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages include:
- `streamlit` - Web framework
- `plotly` - Interactive charts
- `pandas`, `numpy` - Data processing
- `torch`, `transformers` - Deep learning
- `shap`, `lime` - Explainability

### 2. Launch BehterCV

```bash
streamlit run app.py
```

The application will automatically open in your browser at `http://localhost:8501`

---

## 🎯 Features Overview

### 🏠 Dashboard
- **Real-time metrics** with gradient cards
- **Quick demo** with instant analysis
- **Performance charts** showing model accuracy
- **Category distribution** visualization

### 📄 Resume Analyzer
- Upload or paste resume text
- **Instant AI analysis** with confidence scores
- **Skills detection** with professional badges
- **SHAP feature importance** charts
- **Hiring recommendations** from RL agent

### 📊 Batch Processing
- Upload CSV with multiple resumes
- **Process hundreds** of resumes simultaneously
- **Real-time progress** tracking
- **Comprehensive analytics** with charts
- **Export results** to CSV

### 🔍 Smart Search (A*)
- Define job requirements
- **A* algorithm** finds optimal candidates
- **Match scoring** with skills and experience
- **Interactive candidate** profiles
- **Visual comparison** charts

### 🧩 Job Matcher (CSP)
- Define multiple job positions
- **CSP solver** with backtracking + AC-3
- **Optimal assignments** respecting constraints
- **Match quality** visualization
- **Assignment statistics**

### 📈 Analytics
- **Complete model comparison** (Logistic Regression, Random Forest, BERT)
- **Interactive charts** with Plotly
- **Radar charts** for multi-metric comparison
- **Cross-validation** performance
- **Key insights** and recommendations

### ℹ️ About
- Technical architecture details
- Research foundation
- Performance metrics
- Compliance and ethics information

---

## 🎨 UI Highlights

### Modern Design
- **Gradient color scheme** (purple/blue)
- **Professional cards** with hover effects
- **Smooth animations** and transitions
- **Responsive layout** for all screen sizes

### Interactive Charts
- **Plotly visualizations** for real-time interaction
- **Bar charts** for performance comparison
- **Radar charts** for multi-metric analysis
- **Pie charts** for category distribution
- **Line charts** for cross-validation results

### Data Accuracy
- **Real performance metrics** from actual models:
  - BERT: 99.20% accuracy
  - Random Forest: 98.59% accuracy
  - Logistic Regression: 97.79% accuracy
- **Realistic candidate data**
- **Accurate skill matching algorithms**

---

## 💡 Tips for Best Experience

### 1. Resume Analysis
- Paste complete resume text for best results
- Includes more technical skills for accurate categorization
- Review SHAP charts to understand AI reasoning

### 2. Batch Processing
- CSV should have column named: `Resume`, `Resume_str`, `resume`, or `text`
- Preview data before processing
- Download results for record-keeping

### 3. Smart Search
- Enter multiple skills separated by commas
- Adjust experience requirements for better matches
- Review A* algorithm scores for optimal candidates

### 4. Job Matcher
- Define clear skill requirements for each position
- CSP solver finds best non-conflicting assignments
- Check match quality percentages

---

## 🔧 Troubleshooting

### App won't start
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade

# Clear Streamlit cache
streamlit cache clear
```

### Charts not displaying
```bash
# Install plotly
pip install plotly>=5.0.0
```

### Slow performance
- Use Random Forest for faster inference
- Process smaller batches
- Close other browser tabs

---

## 📊 Performance Benchmarks

| Model | Accuracy | F1-Score | Inference Time |
|-------|----------|----------|----------------|
| BERT | 99.20% | 0.9915 | 100ms |
| Random Forest | 98.59% | 0.9858 | 10ms |
| Logistic Regression | 97.79% | 0.9778 | 2ms |

**Throughput:** 14,400 resumes/hour  
**Categories:** 25 job types  
**Explainability:** 100% coverage

---

## 🌟 What's New in BehterCV

### Version 1.0.0
- ✅ Complete UI redesign with modern aesthetics
- ✅ Professional gradient color scheme
- ✅ Interactive Plotly charts
- ✅ Enhanced data accuracy
- ✅ Realistic candidate profiles
- ✅ Improved user experience
- ✅ Better mobile responsiveness
- ✅ Comprehensive analytics dashboard

---

## 📚 Additional Resources

- **GitHub Repository:** https://github.com/ArmanWali/AI-Project
- **Documentation:** See `README.md` for technical details
- **Final Report:** `Final_Report.md` for comprehensive analysis

---

## 🎓 Academic Context

- **Course:** CS-351 Artificial Intelligence
- **Institution:** GIKI
- **Semester:** Fall 2025
- **Project Type:** Final Capstone

---

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the README.md documentation
3. Examine the Final_Report.md for detailed information

---

**Enjoy using BehterCV! 🚀**

*AI-Powered Resume Intelligence Platform*
