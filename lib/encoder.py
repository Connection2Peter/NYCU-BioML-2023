##### Import
import pandas as pd
from Bio import SeqIO
from lib import feature
from lib import dataset



##### Functions
### Encoder
class Encode:
    def __init__(self, db1, db2):
        self.db1 = db1
        self.db2 = db2
    
    def ToOneHot(self):
        X, y = [], []

        db1 = [str(record.seq) for record in SeqIO.parse(self.db1, "fasta")]
        db2 = [str(record.seq) for record in SeqIO.parse(self.db2, "fasta")]

        NewDBs = dataset.Balance(db1, db2)

        del(db1, db2)

        for negativeData in feature.OneHot(NewDBs[0]):
            X.append(negativeData)
            y.append(0)

        for positiveData in feature.OneHot(NewDBs[1]):
            X.append(positiveData)
            y.append(1)

        return pd.DataFrame(X), pd.DataFrame(y)

    def ToAAC(self):
        X, y = [], []

        db1 = [str(record.seq) for record in SeqIO.parse(self.db1, "fasta")]
        db2 = [str(record.seq) for record in SeqIO.parse(self.db2, "fasta")]

        NewDBs = dataset.Balance(db1, db2)

        del(db1, db2)

        for negativeData in feature.AAC(NewDBs[0]):
            X.append(negativeData)
            y.append(0)

        for positiveData in feature.AAC(NewDBs[1]):
            X.append(positiveData)
            y.append(1)

        return pd.DataFrame(X), pd.DataFrame(y)

    def ToPWM(self):
        X, y = [], []

        db1 = [str(record.seq) for record in SeqIO.parse(self.db1, "fasta")]
        db2 = [str(record.seq) for record in SeqIO.parse(self.db2, "fasta")]

        NewDBs = dataset.Balance(db1, db2)

        del(db1, db2)

        for negativeData in feature.PWM(NewDBs[0]):
            X.append(negativeData)
            y.append(0)

        for positiveData in feature.PWM(NewDBs[1]):
            X.append(positiveData)
            y.append(1)

        return pd.DataFrame(X), pd.DataFrame(y)
    
    def ToPSSM(self):
        X, y = [], []

        db1 = [str(record.seq) for record in SeqIO.parse(self.db1, "fasta")]
        db2 = [str(record.seq) for record in SeqIO.parse(self.db2, "fasta")]

        NewDBs = dataset.Balance(db1, db2)

        del(db1, db2)

        for negativeData in feature.PSSM(NewDBs[0]):
            X.append(negativeData)
            y.append(0)

        for positiveData in feature.PSSM(NewDBs[1]):
            X.append(positiveData)
            y.append(1)

        return pd.DataFrame(X), pd.DataFrame(y)

    def ToBLOSUM62(self):
        X, y = [], []

        db1 = [str(record.seq) for record in SeqIO.parse(self.db1, "fasta")]
        db2 = [str(record.seq) for record in SeqIO.parse(self.db2, "fasta")]

        NewDBs = dataset.Balance(db1, db2)

        del(db1, db2)

        for negativeData in feature.BLOSUM62(NewDBs[0]):
            X.append(negativeData)
            y.append(0)

        for positiveData in feature.BLOSUM62(NewDBs[1]):
            X.append(positiveData)
            y.append(1)

        return pd.DataFrame(X), pd.DataFrame(y)