import warnings
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import lightgbm as lgb
import xgboost as xgb
import matplotlib.ticker as ticker
import sys
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression , Ridge , Lasso , BayesianRidge
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn import ensemble
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_log_error
from sklearn import cross_validation
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_digits
from sklearn.learning_curve import learning_curve
from sklearn.cross_validation import ShuffleSplit
from sklearn.grid_search import GridSearchCV
from scipy import stats
from scipy.stats import norm, skew
from scipy.special import boxcox1p
from sklearn.svm import SVR


#sns.set_style('white')
pd.set_option('display.max_columns', None)
base = "./challenge_data/"

# loading the data
houseDataDF = pd.read_csv(base + "train.csv")
idColl = houseDataDF.Id
houseDataDF = houseDataDF.drop(["Id"],axis=1)
houseDataDF.index = idColl
trainSetLength = len(houseDataDF.values)
print(houseDataDF.shape)

testDF = pd.read_csv(base + "test.csv")
testIdColl = testDF.Id
testDF = testDF.drop(["Id"],axis=1)
testDF.index = testIdColl
print(testDF.shape)

usBondsDF = pd.read_csv(base + "T10Y2Y.csv")
usBondsDF.T10Y2Y = usBondsDF.T10Y2Y.apply(lambda x :0.0 if x == "." else float(x)).apply(lambda x : x * 100)
usBondsDF.DATE = usBondsDF.DATE.apply(lambda x : x[:-3])
idColl = usBondsDF.DATE
usBondsDF.index = idColl
usBondsDF = usBondsDF.drop(["DATE"],axis=1)

vacencyDF = pd.read_csv(base + "ILRVAC.csv")
#vacencyDF.ILRVAC = vacencyDF.ILRVAC.apply(lambda x : x * 10)
vacencyDF.DATE = vacencyDF.DATE.apply(lambda x : x[:4])
idColl = vacencyDF.DATE
vacencyDF.index = idColl
vacencyDF = vacencyDF.drop(["DATE"],axis=1)



housingPrisesIndexDF = pd.read_csv(base + "CSUSHPINSA.csv")
housingPrisesIndexDF.CSUSHPINSA_diff = housingPrisesIndexDF.CSUSHPINSA_diff.apply(lambda x :0.0 if x == "NaN" else float(x)).apply(lambda x : x + 3.525000)
housingPrisesIndexDF.DATE = housingPrisesIndexDF.DATE.apply(lambda x : x[:-3])
idColl = housingPrisesIndexDF.DATE
housingPrisesIndexDF.index = idColl
housingPrisesIndexDF = housingPrisesIndexDF.drop(["DATE"],axis=1)
housingPrisesIndexDF.describe()


def findBondsValueByDate(year,month):
    if month < 10:
        month = "0"+str(month)
    date = str(year)+"-"+str(month)
    res = usBondsDF.loc[date][0]
    return res

def findVacencyByYear(year):
    res = vacencyDF.loc[str(year)][0]
    return res

def findHousePriceIndexByDate(year,month):
    if month < 10:
        month = "0"+str(month)
    date = str(year)+"-"+str(month)
    res = housingPrisesIndexDF.loc[date]["CSUSHPINSA_diff"]    
    return res

houseDataDF["USBonds"] = [findBondsValueByDate(x,y) for x,y in zip(houseDataDF.YrSold,houseDataDF.MoSold)]
testDF["USBonds"] = [findBondsValueByDate(x,y) for x,y in zip(testDF.YrSold,testDF.MoSold)]

houseDataDF["Vacency"] = [findVacencyByYear(x) for x in houseDataDF.YrSold]
testDF["Vacency"] = [findVacencyByYear(x) for x in testDF.YrSold]

houseDataDF["HousePriceIndex"] = [findHousePriceIndexByDate(x,y) for x,y in zip(houseDataDF.YrSold,houseDataDF.MoSold)]
testDF["HousePriceIndex"] = [findHousePriceIndexByDate(x,y) for x,y in zip(testDF.YrSold,testDF.MoSold)]

#houseDataDF = houseDataDF.drop(["Price_Range"],axis=1)
corr = houseDataDF.corr()
corrAsList = list(zip(corr.tail(4).values[0],list(corr)))

sortedCorr = sorted(corrAsList,key=lambda x : x[0],reverse=True)

corrDF = pd.DataFrame(data=np.array(sortedCorr),columns=["Correlation","Feature"])
highestCorrFeatures = corrDF["Feature"].values[1:11]

res = houseDataDF.query('SalePrice <= 200000 & OverallQual == 10 ')[highestCorrFeatures]
outliers = res.index.values

houseDataCleanedDF = houseDataDF.drop(outliers)

#Feature engineering

y = np.log(houseDataCleanedDF.SalePrice)
houseDataCleanedDF = houseDataCleanedDF.drop(["SalePrice"],axis=1)
houseDataCleanedDF = pd.concat([houseDataCleanedDF,testDF])

def transformCategoricalFeatures(houseDataDF):
    categorical_feats = houseDataDF.dtypes[houseDataDF.dtypes == "object"].index.values

    for feat in categorical_feats:
        if(feat == "Utilities" or feat == "Alley"):
            continue
            
        dummies = pd.get_dummies(houseDataDF[feat], drop_first=True)
        num_categories = len(dummies.columns)
        houseDataDF[[feat+str(i) for i in range(num_categories)]] = dummies
    
    return houseDataDF
        
    
def makeNewFeatures(houseDataDF):

    # CUSTOM FEATURES
    houseDataDF['NearPark'] = [1 if x=='PosN' or x=='PosA' or y=='PosN' or y=='PosA' else 0\
        for x,y in zip(houseDataDF['Condition1'],houseDataDF['Condition2'])]

    houseDataDF['Loudness'] = [1 if x=='Feedr' or x=='Artery' or x=='RRAe' or y=='Feedr'\
        or y=='Artery' or y=='RRAe' else 0 for x,y in zip(houseDataDF['Condition1'],houseDataDF['Condition2'])]
    
    houseDataDF['TotalSF'] = houseDataDF['TotalBsmtSF'] + houseDataDF['1stFlrSF'] + houseDataDF['2ndFlrSF']
    
    houseDataDF['TimeBetweenRemodAndBuild'] =  [x-y for x,y in zip(houseDataDF["YearRemodAdd"],houseDataDF["YearBuilt"])]
    
    houseDataDF['RemodeledResent'] =  [1 if x != y and 2010 - x < 10 else 0 for x,y in zip(houseDataDF["YearRemodAdd"],houseDataDF["YearBuilt"])]
    
    houseDataDF['Age'] =  [2010 - x for x in houseDataDF["YearBuilt"]]
    
    houseDataDF['AvgQualCond'] =  [(int(x)+int(y))/2 for x,y in zip(houseDataDF["OverallQual"],houseDataDF["OverallCond"])]
    
    houseDataDF['TotalPorchSF'] = houseDataDF['OpenPorchSF'] + houseDataDF['EnclosedPorch'] + houseDataDF['3SsnPorch'] + houseDataDF['ScreenPorch']
    
    houseDataDF['OverallQualSquared'] = [x**2 for x in houseDataDF["OverallQual"]]
    
    houseDataDF['OverallCondSquared'] = [x**2 for x in houseDataDF["OverallCond"]]
    
    houseDataDF['avgQualCondSquared'] = [x**2 for x in houseDataDF["AvgQualCond"]]
    
    houseDataDF['SoldDuringFinancialCrysis'] = [1 if x == 2008 else 0 for x in houseDataDF["YrSold"]]
    
    houseDataDF['NoGarage'] = [1 if x == 0 else 0 for x in houseDataDF["GarageArea"]]
    
    return houseDataDF
    
houseDataCleanedDF = transformCategoricalFeatures(houseDataCleanedDF)
houseDataCleanedDF = makeNewFeatures(houseDataCleanedDF)
# TODO : Add feature for centrality


# Fixing skew

def fixSkew(houseDataDF,verbose=0):
    numeric_feats = houseDataDF.dtypes[houseDataDF.dtypes != "object"].index

    # Check the skew of all numerical features
    skewed_feats = houseDataDF[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    
    skewness = pd.DataFrame({'Skew' :skewed_feats})
    skewness = skewness[abs(skewness) > 0.75]
    
    if verbose != 0:
        print("\nSkew in numerical features: \n")
        print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))


    skewed_features = skewness.index

    lam = 0.15
    
    for feat in skewed_features:
        #all_data[feat] += 1
        if(feat not in ["SalePrice","YearBuilt","HousePriceIndex","Vacency","USBonds"]):#,"HousePriceIndex","Vacency","USBonds"]):
            houseDataDF[feat] = boxcox1p(houseDataDF[feat], lam)
    return houseDataDF

houseDataCleanedDF = fixSkew(houseDataCleanedDF)

# Predictions

data = houseDataCleanedDF.select_dtypes(include=[np.number]).interpolate().dropna()
drop_cols = ["YearBuilt",'YearRemodAdd', 'GarageYrBlt', 'GarageArea', 'TotalBsmtSF', 'BsmtFinSF1']
data = data.drop(drop_cols,axis=1)

columns = data.columns.values

currentTrainSetLenght = trainSetLength - len(outliers)

trainData = pd.DataFrame(data=data.values[:currentTrainSetLenght],columns=columns)

testData = pd.DataFrame(data=data.values[currentTrainSetLenght:],columns=columns)

X = trainData
training_features = X.columns.values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.2)


#Gradient boosting

cv = ShuffleSplit(X_train.shape[0], n_iter=10, test_size=0.2)

def hyperparameterTuning(estimator,param_grid,cv,n_jobs=7): 
    classifier = GridSearchCV(estimator=estimator, cv=cv, param_grid=param_grid, n_jobs=n_jobs,verbose=1)
    classifier.fit(X_train, y_train)
    print("Best Estimator learned through GridSearch")
    print(classifier.best_estimator_)
    return classifier.best_estimator_

arguments = sys.argv[1:]
print(arguments)
# Improving Gradient Boosting
if(arguments[0] == '1'):
    param_grid={'n_estimators':[x for x in range(100,3000,500)],
                'learning_rate': [x/100.0 for x in range(1, 11)],
                'max_depth':[x for x in range(1,11)],
                'min_samples_leaf':[x for x in range(1,11)],
                'max_features':[x/10.0 for x in range(1,11)],
            }

    estimator = GradientBoostingRegressor()
    best_est = hyperparameterTuning(estimator,param_grid,cv)

    pickle.dump(best_est, open("GradientBoostingEstimator_odd.p", "wb"))
elif(arguments[0] == '2'):
    param_grid={'n_estimators':[x for x in range(200,3000,500)],
                'learning_rate': [x/100.0 for x in range(1, 11)],
                'max_depth':[x for x in range(1,11)],
                'min_samples_leaf':[x for x in range(1,11)],
                'max_features':[x/10.0 for x in range(1,11)],
            }

    estimator = GradientBoostingRegressor()
    best_est = hyperparameterTuning(estimator,param_grid,cv)

    pickle.dump(best_est, open("GradientBoostingEstimator_even.p", "wb"))

elif(arguments[0] == '3'):
    param_grid={'learning_rate': [0.05, 0.02, 0.01],
            'max_depth':[4,6,8],
            'gamma':[0.01,0.04,0.08,0.12], 
            'subsample':[0.5,0.75,1],
            'colsample_bytree':[0.5,0.75,1],
            'reg_alpha':[0.001,0.01,0.1],
            'reg_lambda':[0.001,0.01,0.1],
            'n_estimators':[2000,3000]
           }
    
    model_xgb = xgb.XGBRegressor()
    best_est = hyperparameterTuning(model_xgb,param_grid,cv)

    pickle.dump(best_est, open( "XGBoostBestEstimator.p", "wb" ))
elif(arguments[0] == '4'):

    param_grid={'n_estimators': [x for x in range(500,5001,500)],
                'max_bin': [x for x in range(50,150,10)],
                'learning_rate': [x/10.0 for x in range(1,6)],
                'num_iterations': [x for x in range(100,1001,100)],
                'num_leaves': [x for x in range(2,6)],
                'max_depth': [x for x in range(4,13)]
           }
           
    model_lgb = lgb.LGBMRegressor()
    best_est = hyperparameterTuning(model_lgb,param_grid,cv)

    pickle.dump(best_est, open( "LightGBMBestEstimator.p", "wb" ))

else:
    print("ERROR! Invalid job")
