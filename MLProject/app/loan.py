import pandas as pd
import numpy as np
import pickle
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split
import imblearn
from imblearn.over_sampling import SMOTE

df = pd.read_csv("Loan_Data.csv")

#Drop all rows with missing values 
df = df.dropna(axis=0)

#Drop id column
df = df.drop(columns=["Loan_ID"])

#Label Encoding
# label_encoder = preprocessing.LabelEncoder()
# obj = (df.dtypes == "object")

# for col in list(obj[obj].index):
#     df[col] = label_encoder.fit_transform(df[col])

df["Gender"] = df["Gender"].replace(("Male", "Female"),(1,0))
df["Married"] = df["Married"].replace(("Yes", "No"),(1,0))
df["Dependents"] = df["Dependents"].replace(("0","1","2","3+"),(0,1,2,3))
df["Education"] = df["Education"].replace(("Graduate", "Not Graduate"),(1,0))
df["Self_Employed"] = df["Self_Employed"].replace(("Yes", "No"),(1,0))
df["Loan_Status"] = df["Loan_Status"].replace(("Yes", "No"),(1,0))
df["Property_Area"] = df["Property_Area"].replace(("Rural","Semiurban","Urban"),(0,1,2))


#Handle imbalanced data
x= df.drop(["Loan_Status"], axis=1)
y= df["Loan_Status"]

oversample = SMOTE()
x_smote,y_smote = oversample.fit_resample(x,y.values.ravel())

x_train, x_test, y_train, y_test = train_test_split(x_smote,y_smote,test_size=0.2)

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier (n_estimators = 7, criterion = "entropy", random_state = 7)

rfc.fit(x_train, y_train)

# y_pred = rfc.predict(x_train)

pickle.dump(rfc, open("loan.pkl",'wb'))