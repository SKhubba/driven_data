#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 20:11:01 2018

@author: alanzhao

this is a commit
"""

import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

# data directory
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__name__)), 'data')

household_paths = {'A': {'train': os.path.join(DATA_DIR, 'A_hhold_train.csv'),
                         'test': os.path.join(DATA_DIR, 'A_hhold_test.csv')},

                   'B': {'train': os.path.join(DATA_DIR, 'B_hhold_train.csv'),
                         'test': os.path.join(DATA_DIR, 'B_hhold_test.csv')},

                   'C': {'train': os.path.join(DATA_DIR, 'C_hhold_train.csv'),
                         'test': os.path.join(DATA_DIR, 'C', 'C_hhold_test.csv')}}


individual_paths = {'A': {'train': os.path.join(DATA_DIR, 'A_indiv_train.csv'),
                          'test': os.path.join(DATA_DIR, 'A_indiv_test.csv')},

                    'B': {'train': os.path.join(DATA_DIR, 'B_indiv_train.csv'),
                          'test': os.path.join(DATA_DIR, 'B_indiv_test.csv')},

                    'C': {'train': os.path.join(DATA_DIR, 'C_indiv_train.csv'),
                          'test': os.path.join(DATA_DIR, 'C_indiv_test.csv')}}

a_h_train = pd.read_csv(household_paths['A']['train'], index_col='id')
b_h_train = pd.read_csv(household_paths['B']['train'], index_col='id')
c_h_train = pd.read_csv(household_paths['C']['train'], index_col='id')

a_i_train = pd.read_csv(individual_paths['A']['train'], index_col='id')
b_i_train = pd.read_csv(individual_paths['B']['train'], index_col='id')
c_i_train = pd.read_csv(individual_paths['C']['train'], index_col='id')

def standardize(df, numeric_only=True):
    # detect columns that are numeric
    numeric = df.select_dtypes(include=['int64', 'float64'])

    # subtract mean and divide by std
    df[numeric.columns] = (numeric - numeric.mean()) / numeric.std()

    return df


def pre_process_data(df, enforce_cols=None):
    print("Input shape:\t{}".format(df.shape))


    df = standardize(df)
    print("After standardization {}".format(df.shape))

    # create dummy variables for categoricals
    df = pd.get_dummies(df)
    print("After converting categoricals:\t{}".format(df.shape))


    # match test set and training set columns
    if enforce_cols is not None:
        to_drop = np.setdiff1d(df.columns, enforce_cols)
        to_add = np.setdiff1d(enforce_cols, df.columns)

        df.drop(to_drop, axis=1, inplace=True)
        df = df.assign(**{c: 0 for c in to_add})

    df.fillna(0, inplace=True)

    return df

aX_train = pre_process_data(a_h_train.drop('poor', axis=1))
ay_train = np.ravel(a_h_train.poor)

bX_train = pre_process_data(b_h_train.drop('poor', axis=1))
by_train = np.ravel(b_h_train.poor)

cX_train = pre_process_data(c_h_train.drop('poor', axis=1))
cy_train = np.ravel(c_h_train.poor)


