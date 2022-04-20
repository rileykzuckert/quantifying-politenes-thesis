## Preprocessing
# load in libraries and data
import pandas as pd
df = pd.read_csv('ratings.csv', usecols=[0, 1, 3])
df.head()
df.columns = ['phrase', 'patient_ranking', 'patient_sentiment']
df.head()

# extract patient data from dataset
from io import StringIO
col = ['phrase', 'patient_sentiment']
df = df[col]
df.columns = ['phrase', 'patient_sentiment']

# preprocess politeness classes
df['p_sentiment_id'] = df['patient_sentiment'].factorize()[0]
p_sentiment_id_df = df[['patient_sentiment', 'p_sentiment_id']].drop_duplicates().sort_values('p_sentiment_id')
p_sentiment_to_id = dict(p_sentiment_id_df.values)
id_to_category = dict(p_sentiment_id_df[['p_sentiment_id', 'patient_sentiment']].values)
df.head()

print(df.patient_sentiment.unique())
print(df.p_sentiment_id.unique()) # polite = 0, m_impolite = 1,m_polite = 2, neutral = 3

# view bar plot of patient politeness class distributions from survey data
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,6))
df.groupby('p_sentiment_id').patient_sentiment.count().plot.bar(ylim=0)
plt.xlabel('Sentiment', weight='bold')
plt.ylabel('Number of Phrases', weight='bold')
plt.title('Patient Sentiment of Phrases')
plt.xticks([0, 1, 2, 3], ["polite", "moderately impolite", "moderately polite", "neutral"], rotation = 0)
plt.show()

# extract features from text using tf-idf
# calculates a vector for each of the phrases, ex. [0. 0. 0. 1.]
# sublinear_df set to True to use logarithmic form for frequency
# min_df set to 1 for min. # of phrases a word must be present in to be kept - tune this for bigger dataset
# norm set to l2 to ensure all feature vectors have euclidian norm of 1
# ngram_range set to (1, 2) to indicate both unigrams and bigrams considered
# stop_words set to 'english' to remove common pronouns ("a," "the," ...) to reduce noise
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=1, norm='l2', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(df.patient_sentiment).toarray()
labels = df.p_sentiment_id
features.shape
print(features)
  
## Train and test model
# split dataset into train and test sets
# train model
# test model predictions
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

# check class of individual phrases
print(clf.predict(count_vect.transform(["Do you have any issues with gut health, like burping?"])))

## Performance evaluation
# return accuracy from test on test set
sum(predictions == y_test) / len(y_test)

# run benchmark test on 4 classifier models to find best for task
# cross-validate using LOO
# plot avg. accuracy of each model in scatterplot
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

# return model name and average accuracy
cv_df.groupby('model_name').accuracy.mean()

# construct confusion matrix on Naive Bayes model to show discrepancies between predicted and actual labels
conf_mat_df = pd.DataFrame(conf_mat,
                     index = ['polite', 'moderately polite', 'moderately impolite'], 
                     columns = ['polite', 'moderately polite', 'moderately impolite'])
plt.figure(figsize=(3,3))
sns.heatmap(conf_mat_df, annot=True)
plt.title('Confusion Matrix')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
plt.show()

# construct alternative confusion matrix showing percentages of predicted and actual labels
import seaborn as sns
sns.heatmap(conf_mat/np.sum(conf_mat), annot=True, 
            fmt='.2%', cmap='Blues')

# evaluate other metrics of model
from sklearn import metrics
print(metrics.classification_report(y_test, y_pred, labels=[0, 1, 2, 3]))
