import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import AdaBoostClassifier
param_dist_ada = {'algorithm':['SAMME.R','SAMME'],
              'n_estimators':[i for i in range(50,500) if i%10==0],  # (50,100)
              'learning_rate':[0.01,0.05,0.1,0.5,1.0,1.5,2.0]}       # (0.01,2.0)
                            
from sklearn.tree import DecisionTreeClassifier
param_dist_dt = {'criterion':['gini','entropy'],
             'min_samples_split':[i for i in range(2,20)],    # (2,20)
             'min_samples_leaf':[i for i in range(2,20)]}     # (2,20)

from sklearn.neighbors import KNeighborsClassifier
param_dist_knn = {'n_neighbors':[i for i in range(2,20)],     # (2,20)
             'weights':["uniform", "distance"],
             'p':[1,2]}

from sklearn.linear_model import SGDClassifier
param_dist_sgd = {'loss':["hinge", "log", "perceptron"],
             'penalty':["l1", "l2", "elasticnet"],
             'alpha':[0.01,0.03,0.05,0.08,0.1,0.3,0.5,0.8,1.0,1.3,1.5,1.8,2.0]}   # (0.01,2.0)

from sklearn.linear_model import PassiveAggressiveClassifier
param_dist_pac = {'C':[0.01,0.03,0.05,0.08,0.1,0.3,0.5,0.8,1.0,1.3,1.5,1.8,2.0],   # (0.01,2.0)
             'loss':["hinge", "squared_hinge"],
             'tol':[0.01,0.03,0.05,0.08,0.1,0.3,0.5,0.8,1.0,1.3,1.5,1.8,2.0]}      # (0.01,2.0)

from sklearn.naive_bayes import BernoulliNB
param_dist_ber = {'alpha':[0.01,0.05,0.1,0.5,1.0,1.5,2.0], # (0.01,2.0)
             'fit_prior':[True,False]}

from sklearn.tree import ExtraTreeClassifier
param_dist_et = {'criterion':['gini','entropy'],
             'min_samples_split':[i for i in range(2,20)]}   #(2,20)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
param_dist_lda = {'n_components':[i for i in range(1,100) if i%10==0], 
             'tol':[0.001,0.003,0.005,0.008,0.01,0.03,0.05,0.08,0.1]}   # (0.001,0.1)
              
from sklearn.svm import LinearSVC
param_dist_lsvc = {'C':[0.01,0.05,0.1,0.5,1.0,2.0,5.0,8.0,10.0,30.0,50.0,80.0,100.0], # (0.01,100.0)
                   'tol':[0.001,0.005,0.01,0.05,0.1,0.3,0.8,1.0]}  # (0.001,1.0)

from sklearn.svm import SVC
param_dist_svc = {'C':[0.01,0.05,0.1,0.5,1.0,5.0,8.0,10.0,20.0,50.0,80.0,100.0],  # (0.01,100.0)
             'tol':[0.001,0.005,0.01,0.05,0.1,0.5,1.0]}  # (0.001,1.0)
             
from sklearn.naive_bayes import MultinomialNB
param_dist_mnb = {'alpha':[0.01,0.03,0.05,0.08,0.1,0.3,0.5,0.8,1.0,2.0,5.0,8.0,10],  # (0.01,10)
             'fit_prior':[True,False]}

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
param_dist_qda = {'reg_param':[0.0,0.3,0.5,0.8,1.0]}  # (0.0,1.0)
                 
from sklearn.ensemble import RandomForestClassifier
param_dist_rf = {'criterion':["gini", "entropy"],
             'n_estimators':[i for i in range(10,100) if i%10==0]}   # (10,100)
    
#names = ['AdaBoostClassifier','BernoulliNB','DecisionTree','ExtraTree',
#        'LDA','SGD','LSVC','SVC','PAC','RF','KNN'] # without QDA(only 1 hyper parameter)
names = ['LDA','RF']
name_parameter = {'AdaBoostClassifier':param_dist_ada,
                 'BernoulliNB':param_dist_ber,
                 'DecisionTree':param_dist_dt,
                 'ExtraTree':param_dist_et,
                 'KNN':param_dist_knn,
                 'LDA':param_dist_lda,
                 'SGD':param_dist_sgd,
                 'LSVC':param_dist_lsvc,
                 'SVC':param_dist_svc,
                 'MultinomialNB':param_dist_mnb,
                 'PAC':param_dist_pac,
                 'QDA':param_dist_qda,
                 'RF':param_dist_rf}

name_model = {'AdaBoostClassifier':AdaBoostClassifier(),
                 'BernoulliNB':BernoulliNB(),
                 'DecisionTree':DecisionTreeClassifier(),
                 'ExtraTree':ExtraTreeClassifier(),
                 'KNN':KNeighborsClassifier(),
                 'LDA':LinearDiscriminantAnalysis(),
                 'SGD':SGDClassifier(),
                 'LSVC':LinearSVC(),
                 'SVC':SVC(),
                 'MultinomialNB':MultinomialNB(),
                 'PAC':PassiveAggressiveClassifier(),
                 'QDA':QuadraticDiscriminantAnalysis(),
                 'RF':RandomForestClassifier()}

