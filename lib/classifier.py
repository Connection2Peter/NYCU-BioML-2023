##### Import
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier



##### Functions
def DecisionTree():
    return DecisionTreeClassifier(random_state=87)

def RandomForest(nTree):
	return RandomForestClassifier(n_estimators=nTree, random_state=87, n_jobs=-1)

def SupportVectorMachine():
    return SVC(random_state=87)

def XGBoost(nTree):
    return XGBClassifier(n_estimators=nTree, random_state=87, n_jobs=-1)

def MultilayerPerceptron():
    return MLPClassifier(random_state=87, max_iter=2000)
