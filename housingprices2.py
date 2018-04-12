import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

trainy = train['SalePrice']

del train['SalePrice']

alldata = train.append(test)
del alldata['Id']

alldata = pd.get_dummies(alldata)

impute = Imputer()
imputedata = impute.fit_transform(alldata)

colnames = alldata.columns.values
df = pd.DataFrame(imputedata, columns = colnames)

trainx = df[:1460]
testx = df[1460:]

Dtree = DecisionTreeRegressor()
GBoost = GradientBoostingRegressor()
RForest = RandomForestRegressor()
linear = LinearRegression()

GBoost.fit(trainx, trainy)

prediction = GBoost.predict(testx)

final = pd.DataFrame()

ID = pd.read_csv('test.csv')

final['Id'] = ID['Id']
final['SalePrice'] = prediction

final.to_csv('GradientBoosting.csv', index = False)
