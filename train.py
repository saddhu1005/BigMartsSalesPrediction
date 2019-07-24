###############################################################################
# Building the Model

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.externals import joblib
# import pickle

# opening the databases
train_df = pd.read_csv('data/train_data_modified.csv')
test_df = pd.read_csv('data/test_data_modified.csv')

# now let's find mean based prediction
mean_sales = train_df['Item_Outlet_Sales'].mean()

# making baseline models helps in setting a benchmark.
# If the predictive algorithm is below this,
# there is something going seriously wrong with the algorithm or data

baseline = pd.DataFrame({
    'Item_Identifier': test_df['Item_Identifier'],
    'Outlet_Identifier': test_df['Outlet_Identifier'],
    'Item_Outlet_Sales': mean_sales
}, columns=['Item_Identifier', 'Outlet_Identifier', 'Item_Outlet_Sales'])

# Export the baseline result
baseline.to_csv('data/mean_result.csv', index=False)

def trainModel(regressor,   X_train, y_train, filename):
    # fitting the training set on the model
    regressor.fit(X_train, y_train)
    # print the regressor accuracy
    print(filename,"score on training set: %.4f" %(lr.score(X_train,y_train)))
    # Applying K-Fold cross validation
    from sklearn.model_selection import cross_val_score

    accuracies = cross_val_score(estimator=regressor, X=X_train, y=y_train,
                                 cv=10, n_jobs=-1)
    print("mean accuracy of "+filename+" Model on training set: %.4f"
          %(accuracies.mean()))
    print("standard deviation of "+filename+" Model on training set: %.4f"
          %(accuracies.std()))
    # Save the trained model
    joblib.dump(regressor,str("models/"+filename+".sav"))
    # Save the Trained Model
    # pickle.dump(regressor, open("filename.sav", 'wb'))
    # regressor = picle.load(open("filename.sav",'wb'))


# Extracting the training set
X_train = train_df.drop(['Item_Outlet_Sales', 'Item_Identifier',
                         'Outlet_Identifier'], axis=1)
y_train = train_df['Item_Outlet_Sales']

# Linear Regression Model
from sklearn.linear_model import LinearRegression
lr = LinearRegression(normalize=True)

trainModel(lr, X_train, y_train, 'linear_regressor')

# Decision Tree Model
from sklearn.tree import DecisionTreeRegressor
tree = DecisionTreeRegressor(max_depth=9, min_samples_leaf=150)
trainModel(tree, X_train, y_train, 'decision_tree_regressor')

# RandomForest
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=100, n_jobs=-1)
trainModel(lr, X_train, y_train, 'random_forest_regressor')

# SVM
from sklearn.svm import SVR
svr_reg = SVR(kernel='rbf', gamma=0.5)
trainModel(lr, X_train, y_train, 'svm_regressor')


# Gradiant Boosting Regression Model
from sklearn.ensemble import GradientBoostingRegressor
gbr = GradientBoostingRegressor(n_estimators=300, min_samples_leaf=100, max_depth=8)

trainModel(gbr, X_train, y_train, "gradient_boost_regressor")

print("The models are successfully trained")