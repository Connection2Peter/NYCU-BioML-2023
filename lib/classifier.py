##### Import
import joblib
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from catboost import CatBoostClassifier


##### Functions
def Load(path):
    return joblib.load(path)

def Save(model, path):
    joblib.dump(model, path)

def DecisionTree():
    return DecisionTreeClassifier()

def RandomForest(nTree, crit="gini", md=None, mss=2, msl=1, mwfl=0.0):
	return RandomForestClassifier(n_estimators=nTree, criterion=crit, n_jobs=-1)

def SupportVectorMachine():
    return SVC()

def XGBoost(nTree):
    return XGBClassifier(n_estimators=nTree, n_jobs=-1)

def MultilayerPerceptron():
    return MLPClassifier(max_iter=2000)

def VoteClassifier(nTree):
    return VotingClassifier(estimators=[('rf', RandomForest(nTree)), ('rf2', RandomForest(nTree, "entropy")), ('xgb', XGBoost(nTree))], voting='soft')

def CatBoost(nTree):
    return CatBoostClassifier(iterations=nTree, verbose=False)

