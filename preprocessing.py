#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 11:45:12 2019

@author: sadanand
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
train = pd.read_csv('data/Train_data.csv')
test = pd.read_csv('data/Test_data.csv')

# adding source column to differentiate train and test datasets
train['source'] = 'train'
test['source'] = 'test'

# join the data for preprocessing
data = pd.concat([train, test], ignore_index=True, sort=False)

# Handling the missing data
print(data.describe())

# find NaN values
print(data.apply(lambda x: sum(x.isnull())))

# replace the NaN values in "Item_Weight" column with mean
data.fillna({"Item_Weight": data["Item_Weight"].mean()}, inplace=True)

# check the NaN values again
print(data.apply(lambda x: sum(x.isnull())))
# now we have 0 null value in "Item_weight" column
# the nan values present in the data for "Item_outlet_Sales" are for Test_data
# for which we have to predict the outlet_sales.

# column "Outlet_Size" contains nan values but it is catagorical data
# we fill them by "ffill" method i.e. 'forward fill'
# which forwards the last valid observation in place of NaN value.
data["Outlet_Size"].fillna(method="ffill", inplace=True)

# Now all the missing values are handled.

# handling "Item_visibility" of data
# min value is 0 which is unacceptable as all items should be visible to customer
print("Item with Visibility zero :", sum(data["Item_Visibility"] == 0))

visibility_item_avg = data.pivot_table(values="Item_Visibility", index="Item_Identifier")


def impute_visibility_mean(cols):
    visibility = cols[0]
    item = cols[1]
    if visibility == 0:
        return visibility_item_avg['Item_Visibility'][visibility_item_avg.index == item]
    else:
        return visibility


data['Item_Visibility'] = data[['Item_Visibility',
                                'Item_Identifier']].apply(impute_visibility_mean, axis=1).astype(float)

# Outlet_Duration is more importent than Outlet_Establishment_Year
# So create new column for this as the data is of 2013.
data['Outlet_Duration'] = 2013 - data['Outlet_Establishment_Year']

# creating a broad category of Item_Type
# let's Segment Item_Type into 3 category including
# "FD" (Food), "DR" (Drinks) or "NC" (Non-Consumables)
print(data["Item_Type"].value_counts())

# Get the first two chars of ID:
data['Item_Type_Broad'] = data['Item_Identifier'].apply(lambda x: x[0:2])

# Renaming to more intuitive categories:
data['Item_Type_Broad'].replace({'FD': 'Food',
                                 'NC': 'Non-Consumable', 'DR': 'Drinks'}, inplace=True)

print(data["Item_Type_Broad"].value_counts())

# creating a broad category of Item_Fat_Content
print(data["Item_Fat_Content"].value_counts())
data["Item_Fat_Content"].replace({'LF': 'Low Fat',
                                  'low fat': 'Low Fat', 'reg': 'Regular'}, inplace=True)


# Fat_content of non-consumable Item is meaningless
# so convert it to Non-Edible
def impute_Fat_Content(cols):
    Fat = cols[0]
    Item = cols[1]
    if Item == 'Non-Consumable':
        return "Non-Edible"
    else:
        return Fat


data["Item_Fat_Content"] = data[['Item_Fat_Content',
                                 'Item_Type_Broad']].apply(impute_Fat_Content, axis=1)
print(data["Item_Fat_Content"].value_counts())

# Data clean up is done

# Encoding categorical data
# find the unique values
print(data.apply(lambda x: len(x.unique())))
# Encoding the Independent Variable

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
# New variable for outlet
data['Outlet'] = le.fit_transform(data['Outlet_Identifier'])

le = LabelEncoder()

var_mod = ['Item_Fat_Content', 'Outlet_Location_Type',
           'Outlet_Size', 'Item_Type_Broad', 'Outlet_Type', 'Outlet']
for i in var_mod:
    data[i] = le.fit_transform(data[i])

# One Hot Coding:
# Create dummy variables
data = pd.get_dummies(data, columns=['Item_Fat_Content',
                                     'Outlet_Location_Type', 'Outlet_Size',
                                     'Outlet_Type', 'Item_Type_Broad', 'Outlet'])

# avoid dummy value trap
data.drop(['Item_Fat_Content_2', 'Outlet_Location_Type_2', 'Outlet_Size_2',
           'Outlet_Type_3', 'Item_Type_Broad_2', 'Outlet_9'], axis=1, inplace=True)

# Drop the columns which have been converted into different types
data.drop(['Item_Type', 'Outlet_Establishment_Year'], axis=1, inplace=True)

# Split the train and test set
train = data.loc[data['source'] == 'train']
test = data.loc[data['source'] == 'test']

# drop unnecessary columns
train.drop(['source'], axis=1, inplace=True)
test.drop(['source', 'Item_Outlet_Sales'], axis=1, inplace=True)

# Export files as modified versions
train.to_csv("data/train_data_modified.csv", index=False)
test.to_csv("data/test_data_modified.csv", index=False)

print('The Data is now cleaned and modified and preprocessed data is saved')