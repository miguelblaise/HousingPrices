import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

trainy = np.log(train['SalePrice'])
del train['SalePrice']

alldata = train.append(test)

#train.plot.scatter(x = 'LotArea', y = 'LogSalePrice')

del alldata['MiscFeature']
del alldata['Electrical']
del alldata['Heating']
del alldata['RoofMatl']
del alldata['Utilities']
del alldata['Street']
del alldata['Id']
del alldata['PoolQC']

categories = list(alldata.select_dtypes(['object']).columns.values)

alldata['MasVnrType'] = alldata['MasVnrType'].fillna('BrkFace')
alldata['MasVnrArea'] = alldata['MasVnrArea'].fillna(alldata['MasVnrArea'].median())
alldata['BsmtQual'] = alldata['BsmtQual'].fillna('None')
alldata['BsmtCond'] = alldata['BsmtCond'].fillna('None')
alldata['BsmtExposure'] = alldata['BsmtExposure'].fillna('None')
alldata['BsmtFinType1'] = alldata['BsmtFinType1'].fillna('None')
alldata['BsmtFinType2'] = alldata['BsmtFinType2'].fillna('None')
alldata['BsmtFinSF1'] = alldata['BsmtFinSF1'].fillna(alldata['BsmtFinSF1'].median())
alldata['BsmtFinSF2'] = alldata['BsmtFinSF2'].fillna(alldata['BsmtFinSF2'].median())
alldata['BsmtUnfSF'] = alldata['BsmtUnfSF'].fillna(alldata['BsmtUnfSF'].median())
alldata['TotalBsmtSF'] = alldata['TotalBsmtSF'].fillna(0)
alldata['BsmtFullBath'] = alldata['BsmtFullBath'].fillna(0)
alldata['BsmtHalfBath'] = alldata['BsmtHalfBath'].fillna(0)
alldata['KitchenQual'] = alldata['KitchenQual'].fillna('TA')
alldata['FireplaceQu'] = alldata['FireplaceQu'].fillna('None')
alldata['GarageType'] = alldata['GarageType'].fillna('None')
alldata['GarageQual'] = alldata['GarageQual'].fillna('None')
alldata['GarageCond'] = alldata['GarageCond'].fillna('None')
alldata['GarageYrBlt'] = alldata['GarageYrBlt'].fillna(alldata['YearBuilt'])
alldata['GarageFinish'] = alldata['GarageFinish'].fillna('None')
alldata['GarageCars'] = alldata['GarageCars'].fillna(alldata['GarageCars'].median())
alldata['GarageArea'] = alldata['GarageArea'].fillna(alldata['GarageArea'].median())
alldata['Fence'] = alldata['Fence'].fillna('None')
alldata['SaleType'] = alldata['SaleType'].fillna('WD')
alldata['Alley'] = alldata['Alley'].fillna('None')
alldata['LotFrontage'] = alldata['LotFrontage'].fillna(alldata['LotFrontage'].median())
alldata['Exterior1st'] = alldata['Exterior1st'].fillna('VinylSd')
alldata['Exterior2nd'] = alldata['Exterior2nd'].fillna('VinylSd')
alldata['Functional'] = alldata['Functional'].fillna('Typ')

alldata['MSSubClass'] = alldata['MSSubClass'].replace({20: '20', 30: '30', 40: '40', 45: '45', 50: '50', 60: '60', 
                    70: '70', 75: '75', 80: '80', 85: '85', 90: '90', 120: '120', 
                    150: '150', 160: '160', 180: '180', 190: '190'})

alldata['ExterQual'] = alldata['ExterQual'].replace({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0})
alldata['ExterCond'] = alldata['ExterCond'].replace({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0})
alldata['BsmtQual'] = alldata['BsmtQual'].replace({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0})
alldata['BsmtCond'] = alldata['BsmtCond'].replace({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0})
alldata['BsmtExposure'] = alldata['BsmtExposure'].replace({'Gd': 3, 'Av': 2, 'Mn': 1, 'No': 0, 'None': 0})
alldata['HeatingQC'] = alldata['HeatingQC'].replace({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0})
alldata['KitchenQual'] = alldata['KitchenQual'].replace({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0})
alldata['FireplaceQu'] = alldata['FireplaceQu'].replace({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0})
alldata['GarageQual'] = alldata['GarageQual'].replace({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0})
alldata['GarageCond'] = alldata['GarageCond'].replace({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0})
alldata['Functional'] = alldata['Functional'].replace({'Typ': 7, 'Min1': 6, 'Min2': 5, 'Mod': 4, 'Maj1': 3, 'Maj2': 2, 'Sev': 1, 'Sal': 0})
alldata['GarageFinish'] = alldata['GarageFinish'].replace({'Fin': 3, 'RFn': 2, 'Unf': 1, 'None': 0})
alldata['BsmtFinType2'] = alldata['BsmtFinType2'].replace({'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'None': 0})
alldata['BsmtFinType1'] = alldata['BsmtFinType1'].replace({'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'None': 0})

alldata = pd.get_dummies(alldata)

trainx = alldata[:1460]
testx = alldata[1460:]

Dtree = DecisionTreeRegressor()
GBoost = GradientBoostingRegressor()
RForest = RandomForestRegressor()
linear = LinearRegression()

GBoost.fit(trainx, trainy)

prediction = GBoost.predict(testx)

final = pd.DataFrame()

ID = pd.read_csv('test.csv')

final['Id'] = ID['Id']
final['SalePrice'] = np.exp(prediction)

final.to_csv('GradientBoost2.csv', index = False)
