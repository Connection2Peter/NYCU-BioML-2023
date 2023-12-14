##### Import
import pandas as pd
from Bio import SeqIO
from lib import feature
from lib import dataset
from lib import ifeature



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

        y = [0] * len(NewDBs[0]) + [1] * len(NewDBs[1])

        del(db1, db2)

        for data in feature.PWM(NewDBs[0] + NewDBs[1]):
            X.append(data)

        return pd.DataFrame(X), pd.DataFrame(y)
    
    def ToPSSM(self):
        X, y = [], []

        db1 = [str(record.seq) for record in SeqIO.parse(self.db1, "fasta")]
        db2 = [str(record.seq) for record in SeqIO.parse(self.db2, "fasta")]

        NewDBs = dataset.Balance(db1, db2)

        y = [0] * len(NewDBs[0]) + [1] * len(NewDBs[1])

        del(db1, db2)

        for data in feature.PSSM(NewDBs[0] + NewDBs[1]):
            X.append(data)

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
    
    def ToEAAC(self):
        return ifeature.IFeature( self.db1, self.db2, "EAAC").Encode()


### IndependentTest
class IndependentTest:
    def __init__(self, dataset):
        self.dataset = dataset
        self.Kmers = []

    def TSV2Kmers(self, k):
        Seqs = pd.read_csv(self.dataset, sep='\t', header=None).values.tolist()
        Kmer = []

        for Seq in Seqs[1:]:
            for frag in dataset.Seq2Kmer(Seq[1], k):
                Kmer.append(frag)

        self.Kmers = Kmer


    def ToPSSM(self):
        return feature.BLOSUM62(self.Kmers)
