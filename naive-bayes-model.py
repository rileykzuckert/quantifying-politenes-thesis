# https://towardsdatascience.com/multi-class-text-classification-with-scikit-learn-12f1e60e0a9f
import pandas as pd
df = pd.read_csv('ratings.csv', usecols=[0, 1, 2, 3, 4])
df.head()
df.columns = ['phrase', 'patient_ranking', 'clinician_ranking', 'patient_sentiment', 'clinician_sentiment']
df.head()
df.shape
df = df.dropna()
df.shape

from io import StringIO
col = ['phrase', 'patient_sentiment', 'clinician_sentiment']
df = df[col]
df.columns = ['phrase', 'patient_sentiment', 'clinician_sentiment']

# set up patient_sentiment_id
df['p_sentiment_id'] = df['patient_sentiment'].factorize()[0]
p_sentiment_id_df = df[['patient_sentiment', 'p_sentiment_id']].drop_duplicates().sort_values('p_sentiment_id')
p_sentiment_to_id = dict(p_sentiment_id_df.values)
id_to_category = dict(p_sentiment_id_df[['p_sentiment_id', 'patient_sentiment']].values)

# set up clinician_sentiment_id
df['c_sentiment_id'] = df['clinician_sentiment'].factorize()[0]
c_sentiment_id_df = df[['clinician_sentiment', 'c_sentiment_id']].drop_duplicates().sort_values('c_sentiment_id')
c_sentiment_to_id = dict(c_sentiment_id_df.values)
id_to_category = dict(c_sentiment_id_df[['c_sentiment_id', 'clinician_sentiment']].values)

df.head()

print(df.patient_sentiment.unique())
print(df.p_sentiment_id.unique()) # --> polite = 0, m_polite = 1, neutral = 2
print(df.clinician_sentiment.unique())
print(df.c_sentiment_id.unique()) # --> polite = 0, m_polite = 1, neutral = 2

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,6))
df.groupby('p_sentiment_id').patient_sentiment.count().plot.bar(ylim=0)
plt.xlabel('Sentiment', weight='bold')
plt.ylabel('Number of Phrases', weight='bold')
plt.title('Patient Sentiment of Phrases')
plt.xticks([0, 1, 2], ["polite", "moderately polite", "neutral"], rotation = 0)
plt.show()

fig = plt.figure(figsize=(8,6))
df.groupby('c_sentiment_id').clinician_sentiment.count().plot.bar(ylim=0)
plt.xlabel('Sentiment', weight='bold')
plt.ylabel('Number of Phrases', weight='bold')
plt.title('Clinician Sentiment of Phrases')
plt.xticks([0, 1, 2], ["polite", "moderately polite", "neutral"], rotation = 0)
plt.show()

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(df.patient_sentiment).toarray()
labels = df.p_sentiment_id
features.shape

from sklearn.feature_selection import chi2
import numpy as np
N = 2
for p_sentiment, p_sentiment_id in sorted(p_sentiment_to_id.items()):
  features_chi2 = chi2(features, labels == p_sentiment_id)
  indices = np.argsort(features_chi2[0])
  feature_names = np.array(tfidf.get_feature_names())[indices]
  unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
  bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
  print("# '{}':".format(p_sentiment))
  print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
  print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))
  
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
X_train, X_test, y_train, y_test = train_test_split(df['patient_sentiment'], df['p_sentiment_id'], random_state = 0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
clf = MultinomialNB().fit(X_train_tfidf, y_train)

print(clf.predict(count_vect.transform(["Do you have any issues with gut health, like burping?"])))

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
models = [
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(random_state=0),
]
CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []
for model in models:
  model_name = model.__class__.__name__
  accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
  for fold_idx, accuracy in enumerate(accuracies):
    entries.append((model_name, fold_idx, accuracy))
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
import seaborn as sns
sns.boxplot(x='model_name', y='accuracy', data=cv_df)
sns.stripplot(x='model_name', y='accuracy', data=cv_df, 
              size=8, jitter=True, edgecolor="gray", linewidth=2)
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

from IPython.display import display
for predicted in p_sentiment_id_df.p_sentiment_id:
  for actual in p_sentiment_id_df.p_sentiment_id:
    if predicted != actual and conf_mat[actual, predicted] >= 10:
      print("'{}' predicted as '{}' : {} examples.".format(id_to_category[actual], id_to_category[predicted], conf_mat[actual, predicted]))
      display(df.loc[indices_test[(y_test == actual) & (y_pred == predicted)]][['p_sentiment_id', 'patient_sentiment']])
      print('')
      
model.fit(features, labels)
N = 2
for patient_sentiment, p_sentiment_id in sorted(p_sentiment_to_id.items()):
  indices = np.argsort(model.coef_[p_sentiment_id])
  feature_names = np.array(tfidf.get_feature_names())[indices]
  unigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 1][:N]
  bigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 2][:N]
  print("# '{}':".format(patient_sentiment))
  print("  . Top unigrams:\n       . {}".format('\n       . '.join(unigrams)))
  print("  . Top bigrams:\n       . {}".format('\n       . '.join(bigrams)))
  
from sklearn import metrics
print(metrics.classification_report(y_test, y_pred, target_names=df['patient_sentiment'].unique()))
