import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import plotly.express as px
from plotly.offline import iplot

df=pd.read_csv('/kaggle/input/bank-marketing-dataset/bank.csv')
df.head(10)

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import xgboost as xgb
le=LabelEncoder()

non_numeric_cols=[]
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        pass
    else:
        non_numeric_cols.append(col)

for col in non_numeric_cols:
    df[col]=le.fit_transform(df[col])
df.head(10)

x=df.iloc[:,0:16]
y=df['deposit']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)
model = xgb.XGBClassifier(objective='binary:logistic', random_state=42)
model.fit(x_train, y_train)

model.score(x_train, y_train)
model.score(x_test,y_test)
y_predicted = model.predict(x_test)
cm = confusion_matrix(y_test, y_predicted)
cm

print(classification_report(y_test, y_predicted))
print(accuracy_score(y_test,y_predicted))
