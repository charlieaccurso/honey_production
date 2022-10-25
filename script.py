import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

df = pd.read_csv("honeyproduction.csv")

# show head
print(df.head())

# get mean of totalprod per year
prod_per_year= df.groupby('year').totalprod.mean().reset_index()
# print(prod_per_year.head())

# store year column in variable X
X= prod_per_year.year
# reshape X
X= X.values.reshape(-1, 1)

# store totalprod in variable y
y= prod_per_year.totalprod

# scatter plot y by X
plt.scatter(X, y)

# create a linear regression model instance
regr= linear_model.LinearRegression()

# fit the model to the data
regr.fit(X, y)

# print slope and intercept of regression line
print(regr.coef_[0])
print(regr.intercept_)

# store predicted y values
y_predict= regr.predict(X)

# plot regression line
plt.plot(X, y_predict)
plt.show()
plt.clf()

# create a numpy array of future years (x-values)
X_future= np.array(range(2013, 2051))
# reshape the array for sklearn (row to colum)
X_future= X_future.reshape(-1, 1)

# get future y values for X_future
future_predict= regr.predict(X_future)

# plot future_predict (y-value) by X_future on a new plot
plt.plot(X_future, future_predict)
plt.show()

year_2050= X_future[-1].reshape(-1, 1)
print(regr.predict(year_2050)[0])
# according to this model, the value of totalprod will be around 186545 in 2050.
