import numpy as np
import  pandas as pd 
import matplotlib.pyplot as plt
import os


def prepair_missing_data(df):

    for feature in df:
        if df[feature].isnull().sum() > 0:
            if df[feature].dtypes == 'object':
                # Handling missing value for categrical data
                # fill the missing data by most common value
                df[feature].fillna(
                    df[feature].value_counts().index[0], inplace=True)
            elif feature == 'Credit_History':
                df[feature].fillna(method='bfill', inplace=True)

            else:
                # Handling missing value for continuous data
                # fill the missing data with mean
                df[feature].fillna(df[feature].mean(), inplace=True)
    return df


def encoding_categorical_data(df):
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    for feature in df:
        if df[feature].dtypes == 'object':
            df[feature] = encoder.fit_transform(df[feature])
    return df

def one_hot_encoder(df):
    from sklearn.preprocessing import OneHotEncoder
    