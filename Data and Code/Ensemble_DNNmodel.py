# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 19:14:13 2019

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
	model.add(Dense(units = 28, kernel_initializer = 'uniform', activation = 'relu', input_dim = 43))
	model.add(Dense(15, activation='relu'))
	model.add(Dense(2, activation='softmax'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	# fit model
	model.fit(trainX, trainy, epochs=500, verbose=0)
	return model

n_members = 3
for i in range(n_members):
# fit model
    model = fit_model(X_train, y_train)
    	# save model
    filename = 'models\\model' + str(i + 1) + '.h5'
    model.save(filename)
    print('>Saved %s' % filename)


# create directory for models, run only once
#makedirs('models')
#
#	# save first meta model
#model1 = fit_model(x_val1,y_val1)
#    	# save model
#model1.save('models\\model1.h5')
#	# save second meta model
#model2 = fit_model(x_val2,y_val2)
#    
#model2.save('models\\model2.h5')
## save third meta model
#model3 = fit_model(x_train2,y_train2)
#model3.save('models\\model3.h5')





# define stacked model from multiple member input models
def define_stacked_model(members):
	# update all layers in all models to not be trainable
	for i in range(len(members)):
		model = members[i]
		for layer in model.layers:
			# make not trainable
			layer.trainable = False
			# rename to avoid 'unique layer name' issue
			layer.name = 'ensemble_' + str(i+1) + '_' + layer.name
	# define multi-headed input
	ensemble_visible = [model.input for model in members]
	# concatenate merge output from each model
	ensemble_outputs = [model.output for model in members]
	merge = concatenate(ensemble_outputs)
	hidden = Dense(15, activation='relu')(merge)
	output = Dense(2, activation='softmax')(hidden)
	model = Model(inputs=ensemble_visible, outputs=output)
	# plot graph of ensemble
	plot_model(model, show_shapes=True, to_file='model_graph.png')
	# compile
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# fit a stacked model
def fit_stacked_model(model, inputX, inputy):
	# prepare input data
	X = [inputX for _ in range(len(model.input))]
	# encode output data
	
	# fit model
	model.fit(X, inputy, epochs=300, verbose=0)


# make a prediction with a stacked model
def predict_stacked_model(model, inputX):
	# prepare input data
	X = [inputX for _ in range(len(model.input))]
	# make prediction
	return model.predict(X, verbose=0)


# load models from file
def load_all_models(n_models):
	all_models = list()
	for i in range(n_models):
		# define filename for this ensemble
		filename = 'models\\model' + str(i + 1) + '.h5'
		# load model from file
		model = load_model(filename)
		# add to list of members
		all_models.append(model)
		print('>loaded %s' % filename)
	return all_models
# load all models
n_members = 3
members = load_all_models(n_members)
print('Loaded %d models' % len(members))
# define ensemble model
stacked_model = define_stacked_model(members)
# fit stacked model on test dataset
fit_stacked_model(stacked_model, X_test, y_test)

# make predictions and evaluate
y_pred = predict_stacked_model(stacked_model, X_test)
y_pred = (y_pred > 0.5).astype(int) #some of the class multiple class assignments


#y_label=y_test.argmax(1)

# Making the Confusion Matrix

acc = accuracy_score(y_test, y_pred)
print('Stacked Test Accuracy: %.3f' % acc)

from sklearn.metrics import confusion_matrix,classification_report

# With the confusion matrix, we can aggregate model predictions
# This helps to understand the mistakes and refine the model

pred_lbls = np.argmax(y_pred, axis=1)
true_lbls = np.argmax(y_test, axis=1)

cm= confusion_matrix(true_lbls, pred_lbls)
stacked_model.evaluate(X_test, y_test)
target_names =['schizophenia' ,'Normal']
p2=print(classification_report(true_lbls, pred_lbls,target_names=target_names))



