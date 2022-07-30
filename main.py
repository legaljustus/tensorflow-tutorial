import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('Churn.csv')

X = pd.get_dummies(df.drop(['Churn','Customer ID'], axis=1))
y = df['Churn'].apply(lambda x: 1  if x =='Yes' else 0)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.2)

y_train.head()

