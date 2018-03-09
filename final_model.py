import pandas as pd
import numpy as np
import seaborn as sns

from collections import Counter

from catboost import CatBoostClassifier
from catboost import cv
from catboost import Pool

from sklearn.preprocessing import PolynomialFeatures
from itertools import combinations

# import submission function
from data_munge import impute

# import the preprocessed test data
from data_munge import a_h_test, b_i_test, b_h_test, c_h_test

# import the preprocessed data
from data_munge import aX_h_train, ay_h_train, bX_h_train, by_h_train, cX_h_train, cy_h_train
from data_munge import bX_i_train

####################################################################################################

##################################### DATA CLEAN UP ################################################

####################################################################################################

# Returns a bool vector that returns a boolean vector for features with insufficient data

# subject to a threshold 

def Dropcols(df, threshold = 0.02):

    TFvec = np.zeros(len(df.columns), dtype = bool)

    for i in range(len(df.columns)):

        percent_null = sum(df.iloc[: ,i].isnull() ) /len(df.iloc[: ,i])

        if 1- percent_null > threshold:

            TFvec[i] = False

        else:

            TFvec[i] = True

    return (TFvec)


# A function returning location of categorical features

def categorical_indices(input_data):
    categorical_features_indices = np.where(input_data.dtypes != np.float)[0]

    return (categorical_features_indices)


# Another categorical finding function returning boolean vector

def findCat(df):
    TFvec = np.zeros(len(df.columns), dtype=bool)

    for i in range(len(df.columns)):

        if type(df.iloc[1, i]) == str:

            TFvec[i] = True

        else:

            TFvec[i] = False

    return (TFvec)


# Requires findCat, Dropcols, impute

def cleanup(df, df_test, threshold_na=0.10):
    # Dropping features with too much missing data

    drop_vec = Dropcols(df, threshold_na)

    df = df.iloc[:, ~drop_vec]

    df_test = df_test.iloc[:, ~drop_vec]

    # Drop the 'country' variable

    df = df.drop('country', 1)

    df_test = df_test.drop('country', 1)

    # Old stuff

    # Vector for categorical features in df

    cat_vec = findCat(df)

    # Imputing on dummied data frame

    df_temp = pd.get_dummies(df)

    df_temp = impute(df_temp)

    df_test_temp = pd.get_dummies(df_test)

    df_test_temp = impute(df_test_temp)

    # Recombining with original categorical dataset

    df.iloc[:, ~cat_vec] = df_temp.iloc[:, 0:sum(~cat_vec)]

    df_test.iloc[:, ~cat_vec] = df_test_temp.iloc[:, 0:sum(cat_vec)]

    # Return data frames and check

    return ([df, df_test])


# Individual B data, use cleanup function

df_bi = cleanup(bX_i_train, b_i_test)

bX_i_train = df_bi[0]

b_i_test = df_bi[1]

# Household B data, cleanup + changing 1 column with the duplicate name

df_h = cleanup(bX_h_train, b_h_test, threshold_na=0.08)

bX_h_train, b_h_test = df_h[0], df_h[1]

bX_h_train.rename(columns={'wJthinfa': 'wJthinfa1'}, inplace=True)

b_h_test.rename(columns={'wJthinfa': 'wJthinfa1'}, inplace=True)

# Dropping 'country' variable from A and C

aX_h_train = aX_h_train.drop('country', 1)

a_h_test = a_h_test.drop('country', 1)

cX_h_train = cX_h_train.drop('country', 1)

c_h_test = c_h_test.drop('country', 1)

####################################################################################################
############################# INDIVIDUAL FEATURE CREATION ##########################################
####################################################################################################


# Categorical feature vector

feat_b_i = findCat(bX_i_train)

# Continuous: Sum of negative values

b_i_cont = bX_i_train.iloc[:, ~feat_b_i]

b_i_cont = b_i_cont.groupby('id').apply(lambda x: np.sum(x < 0))

b_i_cont_test = b_i_test.iloc[:, ~feat_b_i]

b_i_cont_test = b_i_cont_test.groupby('id').apply(lambda x: np.sum(x < 0))

# Categorical: Most frequent response

b_i_cat = bX_i_train.iloc[:, feat_b_i]

b_i_cat = b_i_cat.groupby(['id']).agg(lambda x: x.value_counts().index[0])

b_i_cat_test = b_i_test.iloc[:, feat_b_i]

b_i_cat_test = b_i_cat_test.groupby(['id']).agg(lambda x: x.value_counts().index[0])

# Reindex train and test to household index

b_i_cont = b_i_cont.reindex(bX_h_train.index.values)

b_i_cat = b_i_cat.reindex(bX_h_train.index.values)

b_i_cont_test = b_i_cont_test.reindex(b_h_test.index.values)

b_i_cat_test = b_i_cat_test.reindex(b_h_test.index.values)

# Final training df: individual data collapsed to household index

b_i_final = pd.concat([b_i_cont, b_i_cat], axis=1)

b_i_final_test = pd.concat([b_i_cont_test, b_i_cat_test], axis=1)

# Feature indices for catboot

b_indices = categorical_indices(b_i_final)

# Indiv model

model_b_i = CatBoostClassifier()

model_b_i.fit(b_i_final, by_h_train, cat_features=b_indices)  # 0.242 best

cv_data_b = cv(params=model_b_i.get_params(), pool=Pool(b_i_final, by_h_train, cat_features=b_indices), iterations=350)

b_score = cv_data_b['Logloss_test_avg'][-1]

# Best features from individual data

best_bi_feats = np.array(model_b_i.get_feature_importance(X=b_i_final, y=by_h_train, cat_features=b_indices)) > 0.35

# Final df to append to household data

b_i_final = b_i_final.iloc[:, best_bi_feats]

b_i_final_test = b_i_final_test.iloc[:, best_bi_feats]

# Combine with household

b_h_combined = pd.concat([bX_h_train, b_i_final], axis=1)

b_h_test_combined = pd.concat([b_h_test, b_i_final_test], axis=1)

####################################################################################################
########################### BUILDING FEATURES ON COMBINED SET ######################################
####################################################################################################

# COUNTRY B

# Fit model

b_indices = categorical_indices(b_h_combined)

model_b_1 = CatBoostClassifier()

model_b_1.fit(b_h_combined, by_h_train, cat_features=b_indices)

cv_data_b1 = cv(params=model_b_1.get_params(), pool=Pool(b_h_combined, by_h_train, cat_features=b_indices),
                iterations=350)

# ~20 best features
# lCKzGQow is most important cat feature

vb_b_feats = np.array(model_b_1.get_feature_importance(X=b_h_combined, y=by_h_train, cat_features=b_indices)) > 1

# Create interactions
train_df_b = b_h_combined.iloc[:, vb_b_feats]

test_df_b = b_h_test_combined.iloc[:, vb_b_feats]

# Lists of categorical/numerical features
char_features_b = list(train_df_b.columns[train_df_b.dtypes == np.object])

num_features_b = list(train_df_b.columns[train_df_b.dtypes == np.float])

# Creating interactions with 2 and 3 variables

char_features_without_lCKzGQow = list(
    train_df_b.columns[(train_df_b.dtypes == np.object) & (train_df_b.columns != 'lCKzGQow')])

cmbs = list(combinations(char_features_b, 2)) + list(
    map(lambda x: ("lCKzGQow",) + x, combinations(char_features_without_lCKzGQow, 2)))


def concat_columns(df, columns):
    value = df[columns[0]].astype(str) + ' '

    for col in columns[1:]:
        value += df[col].astype(str) + ' '

    return value


# Add new features based on combinations/interactions

for cols in cmbs:
    train_df_b["".join(cols)] = concat_columns(train_df_b, cols)

    test_df_b["".join(cols)] = concat_columns(test_df_b, cols)

# Add new engineered features to the list of categorical features in dataframe

add_train_cat_b = train_df_b.iloc[:, sum(vb_b_feats):]

add_test_cat_b = test_df_b.iloc[:, sum(vb_b_feats):]

# Reset training data frame

train_df_b = train_df_b.iloc[:, 0:(sum(vb_b_feats))]

test_df_b = test_df_b.iloc[:, 0:(sum(vb_b_feats))]

# Create continuous interaction terms

num_h_b = train_df_b[num_features_b]

num_h_test_b = test_df_b[num_features_b]

poly = PolynomialFeatures(interaction_only=True, include_bias=False)

# Continuous feature set to add to combined data frame

add_train_num_b = pd.DataFrame(poly.fit_transform(num_h_b), index=train_df_b.index.values).iloc[:, len(num_features_b):]

add_test_num_b = pd.DataFrame(poly.fit_transform(num_h_test_b), index=test_df_b.index.values).iloc[:,
                 len(num_features_b):]

# Final modeling dfs

final_train_b = pd.concat([b_h_combined, add_train_cat_b, add_train_num_b], axis=1)

final_test_b = pd.concat([b_h_test_combined, add_test_cat_b, add_test_num_b], axis=1)

# Fit Model

final_b_ind = categorical_indices(final_train_b)

model_b_final = CatBoostClassifier()

model_b_final.fit(final_train_b, by_h_train, cat_features=final_b_ind)

cv_data_fb = cv(params=model_b_final.get_params(), pool=Pool(final_train_b, by_h_train, cat_features=final_b_ind),
                iterations=350)

# Pick best feats

final_b_feats = np.array(
    model_b_final.get_feature_importance(X=final_train_b, y=by_h_train, cat_features=final_b_ind)) > 0.01

final_train_b = final_train_b.iloc[:, final_b_feats]

final_test_b = final_test_b.iloc[:, final_b_feats]

####################################################################################################

################################## HOUSEHOLD  MODELING #############################################

####################################################################################################


# Country A: The Catboost model, bagging on original household dataset

feature_a = categorical_indices(aX_h_train)

predictions = []

for i in range(10):
    clf = CatBoostClassifier(random_seed=i, logging_level='Silent')

    clf.fit(aX_h_train, ay_h_train, cat_features=feature_a)

    predictions.append(clf.predict_proba(a_h_test)[:, 1])

prediction_a = np.mean(predictions, axis=0)

sns.distplot(prediction_a, bins=15)

# Country B: Catboost, bagging on household dataset with additional features
# from the individual dataset and interaction terms

# Categorical features

feature_b = categorical_indices(final_train_b)

predictions2 = []

for i in range(10):
    clf2 = CatBoostClassifier(random_seed=i, logging_level='Silent')

    clf2.fit(final_train_b, by_h_train, cat_features=feature_b)

    predictions2.append(clf2.predict_proba(final_test_b)[:, 1])

prediction_b = np.mean(predictions2, axis=0)

sns.distplot(prediction_b, bins=15)

# Country C: The Catboost model, bagging on original household dataset

feature_c = categorical_indices(cX_h_train)

predictions3 = []

for i in range(10):
    clf3 = CatBoostClassifier(random_seed=i, logging_level='Silent')

    clf3.fit(cX_h_train, cy_h_train, cat_features=feature_c)

    predictions3.append(clf3.predict_proba(c_h_test)[:, 1])

prediction_c = np.mean(predictions3, axis=0)

sns.distplot(prediction_c, bins=15)

####################################################################################################
#############################################   OUTPUT   ###########################################
####################################################################################################

# Create output

def make_country_sub2(preds, test_feat, country):
    # get just the poor probabilities

    country_sub = pd.DataFrame(data=preds,  # proba p=1

                               columns=['poor'],

                               index=test_feat.index)

    # add the country code for joining later

    country_sub["country"] = country

    return country_sub[["country", "poor"]]


a_sub = make_country_sub2(prediction_a, a_h_test, 'A')

b_sub = make_country_sub2(prediction_b, final_test_b, 'B')

c_sub = make_country_sub2(prediction_c, c_h_test, 'C')

submission = pd.concat([a_sub, b_sub, c_sub])

submission.to_csv('02272018v2.csv')





