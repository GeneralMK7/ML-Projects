# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('email-spam.csv')

# Show first 5 rows (for checking data)
print(data.head(5))

# Convert labels: spam -> 0, ham -> 1
data['Category'] = data['Category'].map({'spam': 0, 'ham': 1})

# Remove rows where Category is missing
data = data.dropna(subset=['Category'])

# Separate features (email text) and labels
X = data['Message']
y = data['Category'].astype(int)

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=3
)

# Convert text into numerical features using TF-IDF
feature_extraction = TfidfVectorizer(
    stop_words='english',     # remove common English words
    lowercase=True,           # convert text to lowercase
    ngram_range=(1, 2),       # use single words and word pairs
    min_df=2,                 # ignore very rare words
    max_features=4000,        # limit number of features
    sublinear_tf=True         # reduce effect of repeated words
)

# Fit TF-IDF on training data and transform it
X_train_features = feature_extraction.fit_transform(X_train)

# Transform test data using same TF-IDF model
X_test_features = feature_extraction.transform(X_test)

# Create Naive Bayes model
model = MultinomialNB(alpha=0.1)

# Train the model
model.fit(X_train_features, y_train)

# Predict on training data
prediction_on_training_data = model.predict(X_train_features)

# Calculate training accuracy
accuracy_on_training_data = accuracy_score(
    y_train, prediction_on_training_data
)

print('Accuracy on training data : ', accuracy_on_training_data)

# Predict on testing data
prediction_on_testing_data = model.predict(X_test_features)

# Calculate testing accuracy
accuracy_on_test_data = accuracy_score(
    y_test, prediction_on_testing_data
)

print('Accuracy on test data : ', accuracy_on_test_data)

# Input email for prediction
input_mail = (
    "This is the 2nd time we are trying to contact you since "
    "you got a free $1000 dollar bike. Claim is easy,call to 777-2222-111"
)

# Convert input email to TF-IDF features
input_data_features = feature_extraction.transform([input_mail])

# Predict whether the email is spam or ham
prediction = model.predict(input_data_features)

print(prediction)

# Print readable output
if prediction[0] == 1:
    print("Model Prediction: HAM (Non-spam email)")
else:
    print("Model Prediction: SPAM (Potentially harmful email)")
