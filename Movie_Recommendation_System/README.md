# 🎬 Movie Recommendation System

A hybrid movie recommendation system built with Python using the [MovieLens dataset](https://grouplens.org/datasets/movielens/). It combines **content-based filtering** (TF-IDF on genres) with a **Streamlit web interface** for interactive recommendations.

---

## 📌 Features

- Content-based filtering using TF-IDF vectorization on movie genres
- Cosine similarity to find movies most similar to a selected title
- Interactive UI built with Streamlit
- Supports 9,700+ movies from the MovieLens dataset

---

## 🗂️ Project Structure

```
movie-recommendation-system/
│
├── Datasets/
│   ├── movies.csv          # Movie metadata (movieId, title, genres)
│   ├── ratings.csv         # User ratings
│   └── tags.csv            # User-assigned tags
│
├── project-test.ipynb      # Data exploration, preprocessing & model building
├── app.py                  # Streamlit frontend
├── movies.joblib           # Preprocessed movies dataframe (saved model artifact)
├── similarity.joblib       # Cosine similarity matrix (large file — see note below)
├── .gitignore
└── README.md
```

---

## ⚙️ How It Works

1. **Data Preprocessing** (in `project-test.ipynb`)
   - Load `movies.csv` and clean genre data (split `|`-separated genres, remove `(no genres listed)`)
   - Convert genre lists to space-separated strings for vectorization

2. **Model Building**
   - Apply `TfidfVectorizer` to the genres column to build a TF-IDF matrix
   - Compute pairwise `cosine_similarity` across all movies
   - Save the matrix and processed dataset using `joblib`

3. **Recommendation Logic** (in `app.py`)
   - Given a movie title, look up its index
   - Sort all movies by cosine similarity score (descending)
   - Return the top N most similar movies (excluding itself)

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/movie-recommendation-system.git
cd movie-recommendation-system
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Generate the similarity matrix

Since `similarity.joblib` is too large for GitHub (see below), regenerate it locally by running all cells in the notebook:

```bash
jupyter notebook project-test.ipynb
```

This will create `similarity.joblib` and `movies.joblib` in your project directory.

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## 📦 Requirements

```
pandas
numpy
scikit-learn
streamlit
joblib
jupyter
```

Install all at once:

```bash
pip install pandas numpy scikit-learn streamlit joblib jupyter
```

---

## ⚠️ About `similarity.joblib` (Large File)

The cosine similarity matrix for ~9,700 movies is a **9708 × 9708 float64 array (~720 MB)**. This exceeds GitHub's 100 MB file size limit and is therefore **not included in this repository**.

**To regenerate it locally**, run the notebook (`project-test.ipynb`) end-to-end. It will automatically save `similarity.joblib` to your project directory.

> **Optional:** If you want to store it in the repo, use [Git LFS](https://git-lfs.github.com/) — see instructions below.

---

## 📁 Dataset

Download the MovieLens dataset from [grouplens.org](https://grouplens.org/datasets/movielens/) and place the CSV files inside a `Datasets/` folder:

- `movies.csv`
- `ratings.csv`
- `tags.csv`

---

## 🙏 Acknowledgements

- [MovieLens](https://grouplens.org/datasets/movielens/) by GroupLens Research
- [scikit-learn](https://scikit-learn.org/) for TF-IDF and cosine similarity
- [Streamlit](https://streamlit.io/) for the web interface