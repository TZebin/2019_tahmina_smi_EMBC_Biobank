# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 16:22:53 2019

@author: User
"""
import numpy as np
import pandas as pd
X_majority=pd.read_csv('F:\\New folder\\control.csv')
X_minority=pd.read_csv('F:\\New folder\\sc_dataset_age.csv')

Combined=pd.concat([X_majority, X_minority],axis=0, ignore_index=True)
X=Combined.iloc[:,0:-1]


y=Combined.iloc[:,42]
print(y.unique())
#y=pd.get_dummies(y,columns=['norm', 'sch'])
from datetime import date

#Convert age
# Age is decode by finding the difference in admission date and date of birth
#    df['age'] = (df['ADMIT_MIN'] - df['DOB']).dt.days // 365
#    df['age'] = np.where(df['age'] < 0, 90, df['age'])

age_ranges = [(40, 60), (60, 70), (70, 100)]
for num, cat_range in enumerate(age_ranges):
    X['Age'] = np.where(X['Age'].between(cat_range[0],cat_range[1]), 
            num, X['Age'])
age_dict = {0: '40-60', 1: '60-70', 2:'senior'}
X['Age'] = X['Age'].replace(age_dict)
X['Age'].describe()
a=pd.get_dummies(X['Age'],columns=['40-60', '60-70','senior'])
X = X.join(pd.DataFrame(a), how='outer')
#
X['31-0.0'].replace({0: 'M', 1:'F'}, inplace=True)
p=pd.get_dummies(X['31-0.0'],columns=['F', 'M'])
X = X.join(pd.DataFrame(p), how='outer')


X=X.join(pd.DataFrame(y), how='outer')
X.to_csv('processed.csv')

#import matplotlib.pyplot as plt
#carrier_count = cat_df_flights['carrier'].value_counts()
#sns.set(style="darkgrid")
#sns.barplot(carrier_count.index, carrier_count.values, alpha=0.9)
#plt.title('Frequency Distribution of Carriers')
#plt.ylabel('Number of Occurrences', fontsize=12)
#plt.xlabel('Carrier', fontsize=12)
#plt.show()



#X.iloc[:,2] = now - X.iloc[:,2]


from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score

x_train, x_val, y_train, y_val = train_test_split(X, y,
                                                  test_size = .3,
                                                  random_state=12)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train)
x_val = sc_X.transform(x_val)

#Counter(y_val)
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=12, ratio = 1.0)
x_train_res, y_train_res = sm.fit_sample(x_train, y_train)

#from collections import Counter
#print('Original dataset shape %s' % Counter(y))
#
##Resample the dataset to handle class imbalance
#from imblearn.over_sampling import SMOTE
#
#
#sm = SMOTE(random_state=42)
#X_res, y_res = sm.fit_resample(X, y)
#print('Resampled dataset shape %s' % Counter(y_res))
#frames2=[pd.DataFrame(X_res), pd.DataFrame(y_res)]
#Combined2=pd.concat(frames2,axis=1)
#Combined2.to_csv('upsampled_smote.csv')
