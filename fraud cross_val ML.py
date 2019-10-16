import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score, cross_val_predict



df=pd.read_csv('fraud_data.csv')

# x: VALUES TO MAKE PREDCITIONS. Y:VALUES TO BE PREDICTED (binary: either 0 or 1)
    #SUPERVISED LEARNING
X=df.iloc[:,:-1]
y=df.iloc[:,-1]

#Let's normalise data values
scaler= MinMaxScaler()
X_scaled = scaler.fit_transform(X)
# visualization of data transformation -> pd.DataFrame(X_scaled)

# WE ARE GOING TO USE SEVERAL MODELS AND THEN COMPARE ACCURACY SCORE
model=RandomForestClassifier()
modelB=LogisticRegression()
modelC=SVC(kernel='rbf')

# Mean value out of the 3 deafult cross_validation folds
mean_val=np.mean(cross_val_score(model, X_scaled,y))
mean_valB=np.mean(cross_val_score(modelB, X_scaled,y))
mean_valC=np.mean(cross_val_score(modelC, X_scaled,y))

print("RandomForestClassifierwe get a cross_val accuracy of {}. ".format(mean_val))
print("LogisticRegression we get a cross_val accuracy of {}. ".format(mean_valB))
print("SVC(kernel='rbf') we get a cross_val accuracy of {}. ".format(mean_valC))

# >>>Output
# RandomForestClassifierwe get a cross_val accuracy of 0.9967270228954992.
# LogisticRegression we get a cross_val accuracy of 0.993730664584454.
# SVC(kernel='rbf') we get a cross_val accuracy of 0.9944682743307481. 
