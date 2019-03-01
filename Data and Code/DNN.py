# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 22:18:49 2019

@author: User
"""

import numpy as np
import pandas as pd
df=pd.read_csv('F:\\New folder\\processed_v1.csv')

df = df.apply(np.random.permutation, axis=0)  
#Load the features
X=df.iloc[:,0:43]
#Load the Labels

y=df.iloc[:,-1]


#Creating a balanced dataset                                                  
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=12, ratio = 1.0)
x_train_res, y_train_res = sm.fit_sample(X, y)



# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_train_res, y_train_res, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


y_train=pd.get_dummies(y_train,columns=['norm', 'sch'])
#original test set (x_val,y_val)
#x_val = sc_X.transform(x_val)
y_test=pd.get_dummies(y_test,columns=['norm', 'sch'])

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout,BatchNormalization

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 18, kernel_initializer = 'uniform', activation = 'relu', input_dim = 43))
classifier.add(Dropout(0.2))

# Adding the second hidden layer
classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(BatchNormalization())

# Adding the output layer
classifier.add(Dense(units = 2, kernel_initializer = 'uniform', activation = 'softmax'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train,epochs=100)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5).astype(int) #some of the class multiple class assignments
y_test=y_test.values

#y_label=y_test.argmax(1)

# Making the Confusion Matrix


from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)
print('Stacked Test Accuracy: %.3f' % acc)

from sklearn.metrics import confusion_matrix,classification_report

# With the confusion matrix, we can aggregate model predictions
# This helps to understand the mistakes and refine the model

pred_lbls = np.argmax(y_pred, axis=1)
true_lbls = np.argmax(y_test, axis=1)

cm= confusion_matrix(true_lbls, pred_lbls)
classifier.evaluate(X_test, y_test)
target_names =['schizophenia' ,'Normal']
p2=print(classification_report(true_lbls, pred_lbls,target_names=target_names))





##Balanced Subclass1: x_val1,y_val1(98 instances)
#x_train1, x_val1, y_train1, y_val1 = train_test_split(x_train_res, y_train_res,
#                                                  test_size = .33,
#                                                  random_state=12)
#
##Balanced Subclass2 and 3: x_val2,y_val2(99 instances), x_train2_y_train2(99 instances)
#
#x_train2, x_val2, y_train2, y_val2 = train_test_split(x_train1, y_train1,
#                                                  test_size = .50,
#                                                  random_state=12)


from sklearn.metrics import accuracy_score
from keras.models import load_model
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model, Sequential
from keras.layers import Input,Dense
import numpy as np
from keras.layers.merge import concatenate

import graphviz
import pydot_ng as pydot
pydot.find_graphviz()
from os import makedirs
from matplotlib import pyplot

def fit_model(trainX, trainy):
	# define model
	model = Sequential()
	model.add(Dense(units = 18, kernel_initializer = 'uniform', activation = 'relu', input_dim = 38))
	model.add(Dense(25, activation='relu'))
	model.add(Dense(2, activation='softmax'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	# fit model
	model.fit(trainX, trainy, epochs=500, verbose=0)
	return model