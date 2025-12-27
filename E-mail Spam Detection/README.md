# 📧 Email Spam Detection using Machine Learning

<div align="center">

  [![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
  [![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
  [![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
  ![Accuracy](https://img.shields.io/badge/Accuracy-99%25-success?style=for-the-badge)
  ![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge)

</div>

## 📖 Description

A machine learning-powered email spam detector that automatically classifies emails as **spam** or **ham** (legitimate). Built using Natural Language Processing (NLP) techniques and the Naive Bayes algorithm, this model achieves an impressive **99% accuracy** on both training and test datasets.

The system analyzes email text content, identifies spam patterns like promotional phrases, urgent calls-to-action, and suspicious language, then predicts whether an email is safe or potentially harmful.

---

## ✨ Features

- 🎯 **High Accuracy** - 99% accuracy on both training and testing data
- ⚡ **Fast Predictions** - Classifies emails in milliseconds
- 🔍 **Smart Text Analysis** - Uses TF-IDF vectorization to understand email content
- 🛡️ **Spam Pattern Detection** - Recognizes common spam phrases like "free offer", "click now", "limited time"
- 🧹 **Text Preprocessing** - Removes stop words and handles keyword stuffing
- 📊 **N-gram Analysis** - Detects phrase patterns (1-2 word combinations)
- 🎨 **Custom Testing** - Easily test your own email messages

---

## 🎬 Project Demo

### Sample Predictions:

**Example 1: Spam Email** ❌
```python
Input: "This is the 2nd time we are trying to contact you since you got a free $1000 dollar bike. Claim is easy, call to 777-2222-111"

Output: [0]
Model Prediction: SPAM (Potentially harmful email)
```

**Example 2: Legitimate Email** ✅
```python
Input: "Hey, can we schedule our project meeting for tomorrow at 3 PM? Let me know if that works."

Output: [1]
Model Prediction: HAM (Non-spam email)
```

### Performance Metrics:
```
Accuracy on training data: 0.99
Accuracy on test data: 0.99
```

---

## 🚀 How to Use

Follow these simple steps to run the spam detector on your machine:

### Step 1: Clone the Repository
```bash
git clone https://github.com/GeneralMK7/ML-Projects.git
cd ML-Projects/email-spam-detection
```

### Step 2: Install Dependencies
Make sure you have Python 3.8+ installed, then install required packages:

```bash
pip install pandas scikit-learn numpy
```

Or use requirements.txt:
```bash
pip install -r requirements.txt
```

### Step 3: Prepare Your Dataset
Ensure `email-spam.csv` is in the same directory with the following structure:
- **Column 1:** `Category` (spam or ham)
- **Column 2:** `Message` (email text content)

### Step 4: Run the Model
```bash
python spam_email-pyfile.py
```

### Step 5: Test Custom Emails
Open `spam_email-pyfile.py` and modify the input_mail variable:

```python
input_mail = "Your custom email text here"
```

Then run the script again to see the prediction!

---

## 🛠️ How It Works

### 1. **Data Loading**
```python
data = pd.read_csv('email-spam.csv')
```
Loads the email dataset containing labeled spam and ham messages.

### 2. **Data Preprocessing**
```python
data['Category'] = data['Category'].map({'spam': 0, 'ham': 1})
data = data.dropna(subset=['Category'])
```
Converts categories to binary (0=spam, 1=ham) and removes null values.

### 3. **Feature Extraction (TF-IDF)**
```python
feature_extraction = TfidfVectorizer(
    stop_words='english',
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_features=4000,
    sublinear_tf=True
)
```
Converts email text into numerical features that the model can understand.

### 4. **Model Training**
```python
model = MultinomialNB(alpha=0.1)
model.fit(X_train_features, y_train)
```
Trains the Naive Bayes classifier on the processed email data.

### 5. **Prediction**
```python
prediction = model.predict(input_data_features)
```
Classifies new emails as spam or ham based on learned patterns.

---

## 📈 Model Performance

| Metric              | Training Data           | Test Data               |
|---------------------|-------------------------|-------------------------|
| **Accuracy**        | 99.12%                  | 98.97%                  |
| **Features Used**   | 4000 TF-IDF features    | 4000 TF-IDF features    |
| **Algorithm**       | Multinomial Naive Bayes | Multinomial Naive Bayes |
| **Prediction Time** | < 1 millisecond         | < 1 millisecond         |

---

## 🎯 Key Parameters

- **alpha=0.1**: Smoothing parameter for Naive Bayes
- **max_features=4000**: Limits vocabulary size to most important words
- **ngram_range=(1,2)**: Considers both single words and two-word phrases
- **min_df=2**: Ignores words appearing in fewer than 2 documents
- **sublinear_tf=True**: Reduces impact of keyword stuffing

---

## 📝 Code Structure

```
email-spam-detection/
│
├── spam_email-pyfile.py    # Main implementation file
├── email-spam.csv           # Dataset with labeled emails
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

---

## 🔧 Customization

### Change the Input Email
Modify line 43 in `spam_email-pyfile.py`:
```python
input_mail = "Your custom email message here"
```

### Adjust Model Parameters
Experiment with different settings:
```python
# Change alpha for smoothing
model = MultinomialNB(alpha=0.5)

# Adjust feature count
max_features=5000

# Try different n-gram ranges
ngram_range=(1, 3)  # Include 3-word phrases
```

---

## 🐛 Troubleshooting

**Issue: Module not found**
```bash
pip install pandas scikit-learn numpy
```

**Issue: CSV file not found**
- Ensure `email-spam.csv` is in the same directory
- Check file name spelling (case-sensitive)

**Issue: Low accuracy**
- Verify dataset has both spam and ham examples
- Check for data preprocessing issues
- Ensure train-test split is working correctly


---

## 📚 Resources & References

- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [Text Classification Tutorial](https://www.kaggle.com/learn/natural-language-processing)
- [Naive Bayes for Spam Filtering](https://en.wikipedia.org/wiki/Naive_Bayes_spam_filtering)
- [TF-IDF Explained](https://www.capitalone.com/tech/machine-learning/understanding-tf-idf/)