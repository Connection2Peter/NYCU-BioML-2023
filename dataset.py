##### Import
import random
import pickle
#import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split



##### Functions
def Seq2Kmer(seq, k):
	RET = []

	for i in range(k, len(seq)-k):
		if seq[i] != "K":
			continue
		
		RET.append(seq[i-k:i+k+1])

	return RET

def Balance_bak(db1, db2):
	return db1, db2

def Balance(db1, db2):
	random.seed(87)
	num1, num2 = len(db1), len(db2)

	if num1 > num2:
		return [random.sample(db1, num2), db2]

	return [db1, random.sample(db2, num1)]

def SplitDataset(X, y, testRatio):
	return train_test_split(X, y, test_size=testRatio, random_state=87)

def SplitNfold(numSplit):
	return StratifiedKFold(n_splits=numSplit, shuffle=True)

def ROC(y_test, y_pred):
	fpr, tpr, _ = roc_curve(y_test, y_pred)

	return [fpr, tpr]

def ROCs(Datas):
	print(Datas)
#	for Data in Datas:
#		plt.plot(Data[1][0], Data[1][1], label='{} (AUC = {})'.format(Data[0], Data[1][2]))
#
#	plt.title("ROC Curve")
#	plt.xlabel("False Positive Rate")
#	plt.ylabel("True Positive Rate")
#	plt.legend(loc=4)
#	plt.show()

def Evaluation(y_test, y_pred):
	tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

	Metrics = [
		tp / (tp + fn),
		tn / (tn + fp),
		accuracy_score(y_test, y_pred),
		matthews_corrcoef(y_test, y_pred),
		roc_auc_score(y_test, y_pred),
	]

	return Metrics

def Normalize2D(Data):
	return MinMaxScaler().fit_transform(Data)

def SaveObject(obj, path):
	with open(path, "wb") as f:
		pickle.dump(obj, f)

def LoadObject(path):
	with open(path, "rb") as f:
		return pickle.load(f)
