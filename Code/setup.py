# -*- coding: utf-8 -*-

from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,PolynomialFeatures

from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


from sklearn.decomposition import PCA,FastICA,KernelPCA


data_preprocessor = [MinMaxScaler,StandardScaler,RobustScaler]
feature_preprocessor = [PCA,FastICA,KernelPCA]
models = [AdaBoostClassifier,DecisionTreeClassifier,SVC]

search_space_dic = {'d1':MinMaxScaler,'d2':StandardScaler,'d3':RobustScaler,
                    'f1':PCA,'f2':FastICA,'f3':KernelPCA,
                    'm1':AdaBoostClassifier,'m2':DecisionTreeClassifier,'m3':SVC
                    }


search_space = '0' * len(search_space_dic)
search_space_values = list(search_space_dic.values())
search_space_names = list(search_space_dic.keys())