# 모든 코드에 있는 라이브러리

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

from collections import Counter

from imblearn.over_sampling import SMOTE

from sklearn.datasets import load_iris, load_diabetes, load_breast_cancer, load_digits, make_classification, make_blobs, make_circles, make_moons

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, silhouette_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge

from sklearn.decomposition import PCA, TruncatedSVD, NMF

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sklearn.svm import SVC, SVR

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from sklearn.neural_network import MLPClassifier, MLPRegressor

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.pipeline import make_pipeline

from sklearn.naive_bayes import GaussianNB

from sklearn.cluster import KMeans, MeanShift, DBSCAN

from sklearn.mixture import GaussianMixture

import xgboost as xgb

import lightgbm as lgb

import re

# 데이터 전처리
---
# 결측값 처리 함수
def handle_missing_values(df):
    for column in df.columns:
        missing_count = df[column].isnull().sum()
        if missing_count > 0:
            # 열의 데이터 타입 확인
            if df[column].dtype in ['float64', 'int64']:
                # 수치형 데이터: 중앙값으로 대체
                df[column].fillna(df[column].median(), inplace=True)
                
                # 평균값
                #df[column].fillna(df[column].mean(), inplace=True)

                #특정 값
                #df[column].fillna(0, inplace=True)  # 0으로 대체

                #평균 값과 중앙값의 평균
                #df[column].fillna((df[column].mean() + df[column].median()) / 2, inplace=True)

            else:
                # 범주형 데이터: 최빈값으로 대체
                df[column].fillna(df[column].mode()[0], inplace=True)

                # 특정 값
                # df[column].fillna('Unknown', inplace=True)

                # 다른열의 값으로 대체(조건부)
                # 예: 특정 기준에 따라 'Alley'의 결측값 대체
                #df[column].fillna(df['Street'].map({'Pave': 'Paved_Alley', 'Grvl': 'Gravel_Alley'}), inplace=True)


                #행 삭제 
                #df.dropna(subset=[column], inplace=True)
    return df

# 처리 전 결측값 확인
print("Before handling missing values:")
print(data.isnull().sum()[data.isnull().sum() > 0])

# 결측값 처리
data = handle_missing_values(data)

# 처리 후 결측값 확인
print("\nAfter handling missing values:")
print(data.isnull().sum()[data.isnull().sum() > 0])
