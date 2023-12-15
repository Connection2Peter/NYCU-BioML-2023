##### Import
import joblib
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from catboost import CatBoostClassifier


##### Functions
def Load(path):
    return joblib.load(path)

def Save(model, path):
    joblib.dump(model, path)

def DecisionTree(**kwargs):
    return DecisionTreeClassifier(**kwargs)

def RandomForest(nTree, n_jobs=-1, **kwargs):
	return RandomForestClassifier(n_estimators=nTree, n_jobs=n_jobs, **kwargs)

def SupportVectorMachine(probability=True, **kwargs):
    return SVC(**kwargs)

def XGBoost(n_estimators, n_jobs=-1, **kwargs):
    return XGBClassifier(n_estimators=n_estimators, n_jobs=n_jobs, **kwargs)

def MultilayerPerceptron(max_iter=2000, **kwargs):
    return MLPClassifier(max_iter=max_iter, **kwargs)

def VoteClassifier(models):
    return VotingClassifier(estimators=models, voting='soft')

def AdaBoost(**kwargs):
    return AdaBoostClassifier(**kwargs)

def GradientBoosting(**kwargs):
    return GradientBoostingClassifier(**kwargs)

def ExtraTrees(**kwargs):
    return ExtraTreesClassifier(**kwargs)

def GaussianNaiveBayes(**kwargs):
    return GaussianNB(**kwargs)

def KNeighbors(**kwargs):
    return KNeighborsClassifier(**kwargs)

def CatBoost(nTree, verbose=False, **kwargs):
    return CatBoostClassifier(iterations=nTree, verbose=verbose, **kwargs)

