import numpy as nm
import matplotlib.pyplot as mtp
import pandas as pd
data = pd.read_csv(r"C:\Users\abhi\Downloads\Rounded_Student_Hours_Studied_vs_Marks_Dataset.csv")
print(data)
x = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)
x_pred = regressor.predict(x_train)


mtp.scatter(x_train,y_train,color = "green")
mtp.plot(x_train,x_pred,color="red")
mtp.title("hours vs marks(training dataset)")
mtp.xlabel("hours studied")
mtp.ylabel("mark scored)")
mtp.show()
