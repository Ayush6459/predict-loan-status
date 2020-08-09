from preprocessing import prepair_missing_data, encoding_categorical_data
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

data=pd.read_csv('loan_status.csv')
X = data.iloc[:, 1:12]
y = data.iloc[:, 12]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=42)

X_train=prepair_missing_data(X_train)
X_test=prepair_missing_data(X_test)

X_train = encoding_categorical_data(X_train)
X_test = encoding_categorical_data(X_test)

encoder=LabelEncoder()

y_train=encoder.fit_transform(y_train)
y_test=encoder.fit_transform(y_test)



model=XGBClassifier()

model.fit(X_train,y_train)

y_pred=model.predict(X_test)

print(roc_auc_score(y_test, y_pred))

model.score(X_test,y_test)