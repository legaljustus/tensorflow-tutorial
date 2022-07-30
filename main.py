import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import keras.api._v2.keras as keras
from keras.models import Sequential, load_model
from keras.layers import Dense
from sklearn.metrics import accuracy_score

df = pd.read_csv('Churn.csv')

X = pd.get_dummies(df.drop(['Churn','Customer ID'], axis=1))
y = df['Churn'].apply(lambda x: 1  if x =='Yes' else 0)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.2)

y_train.head()

model = Sequential()
model.add(Dense(units=32, activation='relu', input_dim=len(X_train.columns)))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='sgd', metrics='accuracy')

model.fit(X_train, y_train, epochs=200, batch_size=32)

y_hat = model.predict(X_test)
y_hat = [0 if val < 0.5 else 1 for val in y_hat]
