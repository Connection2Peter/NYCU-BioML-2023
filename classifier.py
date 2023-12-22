##### Import
import joblib
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


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
	return SVC()

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
