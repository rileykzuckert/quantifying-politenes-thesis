# https://towardsdatascience.com/multi-class-text-classification-with-scikit-learn-12f1e60e0a9f
import pandas as pd
df = pd.read_csv('ratings.csv', usecols=[0, 1, 3])
df.head()
df.columns = ['phrase', 'patient_ranking', 'patient_sentiment']
df.head()

from io import StringIO
col = ['phrase', 'patient_sentiment']
df = df[col]
df.columns = ['phrase', 'patient_sentiment']

# set up patient_sentiment_id
df['p_sentiment_id'] = df['patient_sentiment'].factorize()[0]
p_sentiment_id_df = df[['patient_sentiment', 'p_sentiment_id']].drop_duplicates().sort_values('p_sentiment_id')
p_sentiment_to_id = dict(p_sentiment_id_df.values)
id_to_category = dict(p_sentiment_id_df[['p_sentiment_id', 'patient_sentiment']].values)
df.head()

print(df.patient_sentiment.unique())
print(df.p_sentiment_id.unique()) # --> polite = 0, m_impolite = 1,m_polite = 2, neutral = 3

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,6))
df.groupby('p_sentiment_id').patient_sentiment.count().plot.bar(ylim=0)
plt.xlabel('Sentiment', weight='bold')
plt.ylabel('Number of Phrases', weight='bold')
plt.title('Patient Sentiment of Phrases')
plt.xticks([0, 1, 2, 3], ["polite", "moderately impolite", "moderately polite", "neutral"], rotation = 0)
plt.show()

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=1, norm='l2', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(df.patient_sentiment).toarray()
labels = df.p_sentiment_id
features.shape
print(features)
  
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
X_train, X_test, y_train, y_test = train_test_split(df['patient_sentiment'], df['p_sentiment_id'], random_state = 0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
X_test_counts = count_vect.transform(X_test)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)
clf = MultinomialNB()
clf.fit(X_train_tfidf, y_train)

predictions = clf.predict(X_test_tfidf)
print(predictions)

# test accuracy
sum(predictions == y_test) / len(y_test)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
import seaborn as sns
models = [
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(random_state=0),
]
CV = len(features)
entries = []

loo = LeaveOneOut()
loo.get_n_splits(features)

model_accuracy = {}

for train_index, test_index in loo.split(features):
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    for model in models:
        model_name = model.__class__.__name__
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        if model_name in model_accuracy:
            model_accuracy[model_name].append(sum(predictions == y_test) / len(y_test))
        else:
            model_accuracy[model_name] = [sum(predictions == y_test) / len(y_test)]
cv_accuracy = [[model_name, np.mean(accuracies)] for model_name, accuracies in model_accuracy.items()]
cv_df = pd.DataFrame(cv_accuracy, columns=['model_name', 'accuracy'])
sns.scatterplot(x='model_name', y='accuracy', data=cv_df)
plt.show()

cv_df.groupby('model_name').accuracy.mean()

model = LinearSVC()
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index, test_size=0.33, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import confusion_matrix
conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=p_sentiment_id_df.patient_sentiment.values, yticklabels=p_sentiment_id_df.patient_sentiment.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
