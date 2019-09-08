# -*- coding: utf-8 -*-

from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler

from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


from sklearn.decomposition import PCA,FastICA,KernelPCA

search_space_dic = {'f1':PCA(),'f2':FastICA(),'f3':KernelPCA(),
                    'm1':AdaBoostClassifier()}

search_space = '0' * len(search_space_dic)


