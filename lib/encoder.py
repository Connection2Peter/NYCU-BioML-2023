##### Import
import pandas as pd
from Bio import SeqIO
from lib import feature



##### Functions
### Encoder
class Encoder:
    def __init__(self, positiveDB, negativeDB):
        self.db1 = positiveDB
        self.db2 = negativeDB
    
    def ToOneHot(self):
        X, y = [], []

        for positiveData in feature.OneHot([str(record.seq) for record in SeqIO.parse(self.db1, "fasta")]):
            X.append(positiveData)
            y.append(1)

        for negativeData in feature.OneHot([str(record.seq) for record in SeqIO.parse(self.db2, "fasta")]):
            X.append(negativeData)
            y.append(0)

        return pd.DataFrame(X), pd.DataFrame(y)

    def ToAAC(self):
        X, y = [], []

        for positiveData in feature.AAC([str(record.seq) for record in SeqIO.parse(self.db1, "fasta")]):
            X.append(positiveData)
            y.append(1)

        for negativeData in feature.AAC([str(record.seq) for record in SeqIO.parse(self.db2, "fasta")]):
            X.append(negativeData)
            y.append(0)

        return pd.DataFrame(X), pd.DataFrame(y)

    def ToPWM(self):
        X, y = [], []

        for positiveData in feature.PWM([str(record.seq) for record in SeqIO.parse(self.db1, "fasta")]):
            X.append(positiveData)
            y.append(1)

        for negativeData in feature.PWM([str(record.seq) for record in SeqIO.parse(self.db2, "fasta")]):
            X.append(negativeData)
            y.append(0)

        return pd.DataFrame(X), pd.DataFrame(y)
    
    def ToPSSM(self):
        X, y = [], []

        for positiveData in feature.PSSM([str(record.seq) for record in SeqIO.parse(self.db1, "fasta")]):
            X.append(positiveData)
            y.append(1)

        for negativeData in feature.PSSM([str(record.seq) for record in SeqIO.parse(self.db2, "fasta")]):
            X.append(negativeData)
            y.append(0)

        return pd.DataFrame(X), pd.DataFrame(y)

    def ToBLOSUM62(self):
        X, y = [], []

        for positiveData in feature.BLOSUM62([str(record.seq) for record in SeqIO.parse(self.db1, "fasta")]):
            X.append(positiveData)
            y.append(1)

        for negativeData in feature.BLOSUM62([str(record.seq) for record in SeqIO.parse(self.db2, "fasta")]):
            X.append(negativeData)
            y.append(0)

        return pd.DataFrame(X), pd.DataFrame(y)
