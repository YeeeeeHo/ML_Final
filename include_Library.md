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