import pandas as pd
import re
import joblib
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from utils import preprocessor

data = pd.read_csv('train.csv', names=['sentiment', 'title', 'review'])

X = data.review
y = data.sentiment.replace({1: 'Negative', 2: 'Positive'})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

pipe = Pipeline([
    ('vec', CountVectorizer(stop_words='english', min_df=1000, preprocessor=preprocessor)),
    ('tfid', TfidfTransformer()),
    ('lr', SGDClassifier(loss='log_loss'))
], verbose=True)

model = pipe.fit(X_train, y_train)

joblib.dump(model, 'sentiment_model.joblib')