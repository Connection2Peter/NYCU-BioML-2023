##### Import
import joblib
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

##### Functions
def Load(path):
	return joblib.load(path)

def Save(model, path):
	joblib.dump(model, path)

def DecisionTree():
	return DecisionTreeClassifier()

def RandomForest(nTree):
	return RandomForestClassifier(n_estimators=nTree, n_jobs=-1)

def SupportVectorMachine(probability=True, **kwargs):
	return SVC(**kwargs, probability=probability)

def MultilayerPerceptron():
	return MLPClassifier(max_iter=2000)

def XGBoost(nTree):
	return XGBClassifier(n_estimators=nTree, n_jobs=-1)

def XGBoostHPO():
    param_grid = {
        'reg_alpha': range(0, 1001, 200),
		'max_depth': range(3, 10, 2),
		'learning_rate': [i/100 for i in range(1, 101, 20)],
        'subsample': [i/10 for i in range(5, 10, 2)],
        'n_estimators': range(0, 4001, 500),
    }

    return GridSearchCV(estimator=XGBClassifier(), param_grid=param_grid, cv=5, scoring='roc_auc', n_jobs=-1)

def CatBoost(nTree):
	return CatBoostClassifier(iterations=nTree, verbose=False, random_seed=87)

def VoteClassifier(models):
    return VotingClassifier(estimators=models, voting='soft')

def StackClassifier(models):
    return StackingClassifier(estimators=models)


def AdaBoost(**kwargs):
    return AdaBoostClassifier(**kwargs)

def GradientBoosting(**kwargs):
    return GradientBoostingClassifier(**kwargs)

def ExtraTrees(nTree, **kwargs):
    return ExtraTreesClassifier(n_estimators=nTree, **kwargs)

def GaussianNaiveBayes(**kwargs):
    return GaussianNB(**kwargs)

def KNeighbors(**kwargs):
    return KNeighborsClassifier(**kwargs)

def LightGBM(**kwargs):
		return LGBMClassifier(**kwargs)
