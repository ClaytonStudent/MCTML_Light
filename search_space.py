import copy
# Classifier 
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

# Data Preprocessing (5)
from sklearn.preprocessing import Normalizer,MinMaxScaler,QuantileTransformer,StandardScaler,RobustScaler

# Feature Preprocessing (8)
from sklearn.decomposition import FastICA
from sklearn.cluster import FeatureAgglomeration
from sklearn.decomposition import KernelPCA
from sklearn.kernel_approximation import RBFSampler
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures

from itertools import product
def smart_init(hyper_parameters):
    hyper_parameters = copy.deepcopy(hyper_parameters)
    if hyper_parameters == None:
        return None
    for name,value in hyper_parameters.items():
        if type(value) == tuple:
            init_value = []
            low,high = value[0],value[1]
            ratio = (high-low)/4
            for i in range(5):
                if type(low) == int:
                    init_value.append(int(low+ratio*i))
                else:
                    init_value.append(low+ratio*i)
            hyper_parameters[name] = init_value
    return [tuple(zip(hyper_parameters, v)) for v in product(*hyper_parameters.values())] 

class BasePerprocessor():
    def __init__(self,name=None,model=None,hyper_parameters=None):
        self.name = name
        self.model = model
        self.hyper_parameters = hyper_parameters
        self.search_space = smart_init(self.hyper_parameters)
        
    def update(self,hypers):
        #self.model = self.model.set_params(**hypers)
        self.model.set_params(**hypers)
        
    def fit_transform(self,X,y):
        self.model.fit_transform(X,y)
    
    def fit(self,X,y):
        self.model.fit(X,y)
    
    def predict(self,X):
        self.model.predict(X)

## Data Preprocessing
quantile = BasePerprocessor(name='norm',model=QuantileTransformer(),hyper_parameters={'n_quantiles':(10,2000),'output_distribution':['uniform','normal']})
norm = BasePerprocessor(name='norm',model=Normalizer(),hyper_parameters={'norm':['l1','l2','max']})
minmax = BasePerprocessor(name='MinMax',model = MinMaxScaler(),hyper_parameters={'feature_range': [(0, 1)]})
robust = BasePerprocessor(name='robust',model=RobustScaler(),hyper_parameters={'quantile_range': [(25.0, 75.0),(30.0,70.0)]})
stand = BasePerprocessor(name='StandardScaler',model =StandardScaler(),hyper_parameters=None)
        
## Feature Preprocessing
fastica = BasePerprocessor(name='FastICA',model=FastICA(),hyper_parameters={'n_components':(10,2000),'algorithm':['parallel','deflation'],'fun':['logcosh','exp','cube']})

pca = BasePerprocessor(name='PCA',model = PCA(),hyper_parameters={'tol':(0.01,0.1)})

polynomial = BasePerprocessor(name='PolynomialFeatures', model=PolynomialFeatures(),hyper_parameters={'degree':[2,3],
'interaction_only':[False,True],'include_bias':[True,False]})

feature_agglomeration = BasePerprocessor(name='FeatureAgglomeration',model=FeatureAgglomeration(),hyper_parameters={'n_clusters':(2,35)})

kernel_pca = BasePerprocessor(name='KernelPCA',
                                     model=KernelPCA(),
                                     hyper_parameters={'n_components':(10,2000),
                                                       'kernel':['poly','rbf','sigmoid','cosine']})
        
## Classifiers 
knn = BasePerprocessor(name='KNN',model=KNeighborsClassifier(),
                       hyper_parameters={'n_neighbors':(2,20),
                                         'weights':['uniform','distance'],
                                         'p':[1,2]})

decision_tree = BasePerprocessor(name='DecisionTree',
                                 model = DecisionTreeClassifier(),
                                 hyper_parameters={'criterion':['gini','entropy'],
                                                   'min_samples_split':(2,20),
                                                   'min_samples_leaf':(2,20)})

extra_tree = BasePerprocessor(name='ExtraTree',
                              model=ExtraTreeClassifier(),
                              hyper_parameters={'criterion':['gini','entropy'],
                                                'min_samples_split':(2,20)})

ada = BasePerprocessor(name='Ada',model=AdaBoostClassifier(),
                       hyper_parameters={'algorithm':["SAMME.R", "SAMME"],
                                         'n_estimators':(50,500),
                                         'learning_rate':(0.01,2)})

bernou = BasePerprocessor(name='BernoulliNe',
                          model= BernoulliNB(),
                          hyper_parameters= {'alpha':(0.01,100),
                                             'fit_prior':[True,False]})


gaussian_nb = BasePerprocessor(name='GaussianNB',
                               model=GaussianNB(),
                               hyper_parameters={'var_smoothing':( 1e-09, 1e-07)})

gradient_boost = BasePerprocessor(name='GradientBoosting',
                                  model=GradientBoostingClassifier(),
                                  hyper_parameters={'learning_rate':(0.01,1),
                                                    'max_iter':(32,512),
                                                    'min_samples_leaf':(1,200),
                                                    'max_leaf_nodes':(3,2047)})

lda = BasePerprocessor(name='lda',
                       model=LinearDiscriminantAnalysis(),
                       hyper_parameters={'shrinkage':["None", "auto", "manual"],
                                         'n_components':(1,250),
                                         'tol':(1e-5, 1e-1)})

lsvc = BasePerprocessor(name='lsvc',
                         model=LinearSVC(),
                         hyper_parameters={'C':(0.03,32768),
                                           'tol':(1e-5, 1e-1)})

svc =BasePerprocessor(name='svc',
                      model=SVC(),
                      hyper_parameters={'C':(0.03,32768),
                                        'kernel':["rbf", "poly", "sigmoid"],
                                           'tol':(1e-5, 1e-1)})

mnb = BasePerprocessor(name='MultinomialNB',
                       model=MultinomialNB(),
                       hyper_parameters={'alpha':(1e-2,100),
                                         'fit_prior':[True,False]})

passiv_aggressive = BasePerprocessor(name='PassiveAggressiveClassifier',
                                     model=PassiveAggressiveClassifier(),
                                     hyper_parameters={'C':(1e-5, 10),
                                                       'loss':["hinge", "squared_hinge"],
                                                       'tol':(1e-5, 1e-1),
                                                       'average':[True,False]})

qda = BasePerprocessor(name='QDA',
                       model=QuadraticDiscriminantAnalysis(),
                       hyper_parameters={'tol':(0.0001,0.1),
                                         'reg_param':(0.0,1.0)})

rf = BasePerprocessor(name='RF',
                      model=RandomForestClassifier(),
                      hyper_parameters={'n_estimators':(10,100),
                                        'criterion':["gini", "entropy"],
                                        'max_features':(0.1,1.0),
                                        'min_samples_split':(2,20)})




data_actions = ['quantile','robust','norm','minmax']
feature_actions = ['kernel_pca','polynomial','pca','fastica']
classifier_actions = ['knn','decision_tree','extra_tree','ada','bernou','gaussian_nb',
                      'gradient_boost','lda','lsvc','svc','mnb','passiv_aggressive','qda','rf']

check_list = {'quantile':quantile,'robust':robust,'norm':norm,'minmax':minmax,
              'kernel_pca':kernel_pca,'polynomial':polynomial,'pca':pca,'fastica':fastica,
              'knn':knn,'decision_tree':decision_tree,'extra_tree':extra_tree,'ada':ada,'bernou':bernou,
              'gaussian_nb':gaussian_nb,'gradient_boost':gradient_boost,'lda':lda,'lsvc':lsvc,'svc':svc,
              'mnb':mnb,'passiv_aggressive':passiv_aggressive,'qda':qda,'rf':rf
             }
             
        
        
