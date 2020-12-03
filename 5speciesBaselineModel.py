# Import required libraries
import pandas as pd
import numpy as np 
import sklearn
from sklearn.model_selection import train_test_split
# Keras specific
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical 

# read in data

df = pd.read_csv("./crossfeeding/5fcs/fiveS.csv")
df = df.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'])
spec_nums = {"AA":1,"OD":2,"PC":3,"SR":4,"VG":5}
df["spec_num"] = [spec_nums[i] for i in df["species"]]
#pre-process
target_column = ['spec_num'] # response varable
predictors = list(set(list(df.columns))-set(["Width",
"Time","well","dilution","media","copper","species","spec_num"])) # predictors # predictors
df[predictors] = df[predictors]/df[predictors].max()# normalise predictors
df.describe()
df = df.sample(frac=1) #shuffle rows

# split data
X = df[predictors].values
y = df[target_column].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40)
print(X_train.shape); print(X_test.shape)
#specify that response is categorical
y_train = to_categorical(y_train)[:,1:]
y_test = to_categorical(y_test)[:,1:]
count_classes = y_test.shape[1]
print(count_classes)

# define model
model = Sequential()
model.add(Dense(12, activation='relu', input_dim=X.shape[1]))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(5, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])


# build the model
model.fit(X_train, y_train, epochs=80)

#evaluate the model
pred_train= model.predict(X_train)
scores = model.evaluate(X_train, y_train, verbose=0)
print('Accuracy on training data: {}% \n Error on training data: {}'.format(scores[1], 1 - scores[1]))   
 
pred_test= model.predict(X_test)
scores2 = model.evaluate(X_test, y_test, verbose=0)
print('Accuracy on test data: {}% \n Error on test data: {}'.format(scores2[1], 1 - scores2[1]))    

predictions = model.predict(X_test)

y_df = pd.DataFrame(y_test)
pred_df = pd.DataFrame(predictions)
y_df.to_csv("5y.csv")
pred_df.to_csv("5pred.csv")