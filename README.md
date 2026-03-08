# 🤖 Machine Learning Projects
Welcome to my machine learning projects repository! This is where I document my learning journey
as I explore the exciting world of artificial intelligence and data science. Each project here represents a
step forward in understanding how machines can learn from data.

## 📚 Projects

### 🎬 Movie Recommendation System
**Algorithm:** Content-Based Filtering (TF-IDF + Cosine Similarity)  
**Dataset:** MovieLens (9,708 movies)

Recommends movies based on genre similarity using TF-IDF vectorization and cosine similarity, deployed as an interactive Streamlit web app.

**Features:**
- TF-IDF vectorization on 19 genre tokens
- Pairwise cosine similarity across 9,708 movies
- Top-5 recommendations per selected movie
- Interactive Streamlit UI with dropdown selection
- Model artifacts serialized with joblib

**Tech Stack:** Python, Pandas, Scikit-learn, Streamlit, Joblib

📖 [Detailed guide](https://github.com/GeneralMK7/ML-Projects/blob/master/Movie_Recommendation_System/) available in project folder

> ⚠️ `similarity.joblib` (~720 MB) is excluded from the repo. Run `project-test.ipynb` end-to-end to regenerate it locally.

---

### 📧 Email Spam Detection
**Algorithm:** Multinomial Naive Bayes  
**Accuracy:** 99.3% (training), 98.6% (testing)

Classifies emails as spam or ham using TF-IDF text vectorization and Naive Bayes classification.

**Features:**
- TF-IDF vectorization with 4000 features
- N-gram analysis (1,2) for phrase detection
- Stop words removal
- Handles keyword stuffing

**Tech Stack:** Python, Pandas, Scikit-learn

📖 [Detailed guide](https://github.com/GeneralMK7/ML-Projects/blob/master/E-mail%20Spam%20Detection/) available in project folder

---

## 🛠️ Technologies
- **Python 3.x**
- **Pandas** - Data manipulation
- **Scikit-learn** - ML algorithms (TF-IDF, Cosine Similarity, Naive Bayes)
- **NumPy** - Numerical computing
- **Streamlit** - Web app deployment
- **Joblib** - Model serialization

## 🚀 Setup
```bash
# Clone repository
git clone https://github.com/GeneralMK7/ML-Projects.git
cd ML-Projects

# Install dependencies
pip install -r requirements.txt

# Navigate to any project folder and follow its README
cd project-name/
```

## 📂 Structure
```
ml-projects/
├── Movie-Recommendation-System/
│   ├── project-test.ipynb
│   ├── app.py
│   ├── movies.joblib
│   ├── similarity.joblib       # excluded from repo — regenerate via notebook
│   ├── Datasets/
│   │   ├── movies.csv
│   │   ├── ratings.csv
│   │   └── tags.csv
│   └── README.md
├── email-spam-detection/
│   ├── spam_email-pyfile.py
│   ├── email-spam.csv
│   └── README.md
└── requirements.txt
```

## 📊 Results

**Movie Recommendation System:**
- Dataset: 9,708 movies across 19 genres
- Similarity matrix: 9,708 × 9,708 cosine scores
- Response time: < 1ms per recommendation

**Email Spam Detection:**
- Training: 99% accuracy
- Testing: 98% accuracy
- Prediction time: < 1ms

## 📫 Contact
- GitHub: [@GeneralMK7](https://github.com/GeneralMK7)
- LinkedIn: [MadhuKiranGolla](https://www.linkedin.com/in/golla-madhu-kiran-6b5a1b322/)
- Email: madhukiran2k6@gmail.com
