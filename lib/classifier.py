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

def DecisionTree():
    return DecisionTreeClassifier()

def RandomForest(nTree):
	return RandomForestClassifier(n_estimators=nTree, n_jobs=-1)

def SupportVectorMachine():
    return SVC(probability=True)

def XGBoost(nTree):
    return XGBClassifier(n_estimators=nTree, n_jobs=-1)

def MultilayerPerceptron():
    return MLPClassifier(max_iter=2000)

def VoteClassifier(nTree, models):
    return VotingClassifier(estimators=models, voting='soft')

def AdaBoost():
    return AdaBoostClassifier()

def GradientBoosting():
    return GradientBoostingClassifier()

def ExtraTrees():
    return ExtraTreesClassifier()

def GaussianNaiveBayes():
    return GaussianNB()

def KNeighbors():
    return KNeighborsClassifier()

def CatBoost(nTree):
    return CatBoostClassifier(iterations=nTree, verbose=False)

