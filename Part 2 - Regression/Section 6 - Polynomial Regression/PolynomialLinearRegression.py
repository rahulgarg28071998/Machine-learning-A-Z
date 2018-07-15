import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#dataset
dataset = pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[: , 1:2].values
Y=dataset.iloc[:,2].values

#linear regression model
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)
 
#polynomial regression model
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=5)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,Y)

#linear visual
plt.scatter(X,Y,color='red')
plt.plot(X,lin_reg.predict(X),color='blue')
plt.plot(X,lin_reg_2.predict(poly_reg.fit_transform(X)),color='green')

#
lin_reg.predict(6.5)
lin_reg_2.predict(poly_reg.fit_transform(6.5))