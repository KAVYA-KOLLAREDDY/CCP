import pandas as pd
covid = pd.read_csv('E:\covid-data.csv')
india_case=covid[covid["location"]=="India"]
import seaborn as sns
from matplotlib import pyplot as plt
sns.set(rc={'figure.figsize':(15,10)})
sns.lineplot(x="date",y="total_cases",data=india_case)
plt.show()
india_last_5_days=india_case.tail()
sns.set(rc={'figure.figsize':(10,5)})
sns.lineplot(x="date",y="total_cases",data=india_last_5_days)
plt.show()
from sklearn.model_selection import train_test_split
import datetime as dt
india_case['date'] = pd.to_datetime(india_case['date']) 
india_case.head()
india_case['date']=india_case['date'].map(dt.datetime.toordinal)
india_case.head()
x=india_case['date']
y=india_case['total_cases']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)  
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
import numpy as np
lr.fit(np.array(x_train).reshape(-1,1),np.array(y_train).reshape(-1,1))
india_case.tail()
y_pred=lr.predict(np.array(x_test).reshape(-1,1))
from sklearn.metrics import mean_squared_error
mean_squared_error(x_test,y_pred)
lr.predict(np.array([[738021]]))

