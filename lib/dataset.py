##### Import
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold



##### Functions
def SplitNfold(numSplit):
	return StratifiedKFold(n_splits=numSplit, shuffle=True, random_state=87)

def Accuracy(y_test, y_pred):
	return accuracy_score(y_test, y_pred)

