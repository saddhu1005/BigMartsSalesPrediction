########################################
# Running the model on test dataset and saving the predictions
import numpy as np
import matplotlib.pyplot as mp
import pandas as pd
from sklearn.externals import joblib

# opening the test databases
# train_df = pd.read_csv('data/train_data_modified.csv')
test_df = pd.read_csv('data/test_data_modified.csv')

# Creating the test set
X_test = test_df.drop(['Item_Identifier', 'Outlet_Identifier'], axis=1)

# Function to predict and save predictions of different regressors
def testModel(filename):
    # Load the saved Model
    regressor = joblib.load(str("models/"+filename+".sav"))

    # Predict the results
    y_pred = regressor.predict(X_test)

    # Formulate and Export rhe result
    result = pd.DataFrame({
            'Item_Identifier': test_df['Item_Identifier'],
            'Outlet_Identifier': test_df['Outlet_Identifier'],
            'Item_Outlet_Sales': y_pred},
    columns=['Item_Identifier', 'Outlet_Identifier', 'Item_Outlet_Sales'])
    
    result.to_csv(str('data/'+filename+'_result.csv'), index=False)
    
    print(filename, ' Model run successfully on test set and predictions are saved')

testModel('linear_regressor')
testModel('decision_tree_regressor')
testModel('random_forest_regressor')
testModel('svm_regressor')
testModel('gradient_boost_regressor')
