import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

fake_news = pd.read_csv('Project_2_false.csv')
true_news = pd.read_csv('Project_2_true.csv')

fake_news['label'] = 0
true_news['label'] = 1

df = pd.concat([fake_news, true_news])

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=50)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

coefficients = model.coef_
feature_names = vectorizer.get_feature_names_out()
coef_df = pd.DataFrame({'word': feature_names, 'coefficient': coefficients[0]})
coef_df = coef_df.sort_values(by='coefficient', key=abs, ascending=False)
print(coef_df)