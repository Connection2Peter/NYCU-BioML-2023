##### Import
import os
import tempfile
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

        self.do_PWM_p   = False
        self.do_PWM_n   = False
        self.do_PWM_all = False
        self.do_PWM_d   = False
        self.do_PWM_d2  = False
        self.do_PWM_d3  = False

        self.PWM_p   = None
        self.PWM_n   = None
        self.PWM_all = None
        self.PWM_d   = None
        self.PWM_d2  = None
        self.PWM_d3  = None

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

    def _iFeatureEncode(self, encoder):
        db1 = [str(record.seq) for record in SeqIO.parse(self.db1, "fasta")]
        db2 = [str(record.seq) for record in SeqIO.parse(self.db2, "fasta")]

        NewDBs = dataset.Balance(db1, db2)

        total_fasta = ""
        for i, seq in enumerate(NewDBs[0]):
            total_fasta += f">Pos{i}\n" + seq + "\n"

        for i, seq in enumerate(NewDBs[1]):
            total_fasta += f">Neg{i}\n" + seq + "\n"

        db1_len = len(NewDBs[0])
        db2_len = len(NewDBs[1])

        temp_dir = "bin/iFeature/tmp"
        os.makedirs(temp_dir, exist_ok=True)

        temp_fastaFile = tempfile.NamedTemporaryFile(dir=temp_dir).name + "_total.fasta"

        try:
            fwrite = open(temp_fastaFile, "w")
            fwrite.write(total_fasta)
            fwrite.close()
        except:
            print("Error: Cannot write file " + temp_fastaFile)
            return

        temp_out = temp_fastaFile[:-12] + "_" + encoder + ".csv"

        cmd = f"python bin/iFeature/iFeature.py --file {temp_fastaFile} --type {encoder} --out {temp_out}"

        try:
            os.system(cmd)
        except:
            print("Error: Cannot execute command " + cmd)
            return

        try:
            os.remove(temp_fastaFile)
        except:
            print("Error: Cannot remove file " + temp_fastaFile)
            return

        X, y = [], []
        try:
            content = pd.read_csv(temp_out, sep='\t')
        except:
            print("Error: Cannot open file " + temp_out)
            return

        for i in range (db1_len + db2_len):
            X.append( list(content.iloc[i, 1:]) )

        y = [0] * db1_len + [1] * db2_len

        try:
            os.remove(temp_out)
        except:
            print("Error: Cannot remove file " + temp_out)
            return

        return pd.DataFrame(X), pd.DataFrame(y)

    def ToKSCTriad(self):
        return self._iFeatureEncode("KSCTriad")

    def ToCTriad(self):
        return self._iFeatureEncode("CTriad")

    def ToCTDD(self):
        return self._iFeatureEncode("CTDD")

    def ToZSCALE(self):
        return self._iFeatureEncode("ZSCALE")

    def ToGTPC(self):
        return self._iFeatureEncode("GTPC")

    def ToGDPC(self):
        return self._iFeatureEncode("GDPC")

    def ToEGAAC(self):
        return self._iFeatureEncode("EGAAC")

    def ToBINARY(self):
        return self._iFeatureEncode("BINARY")

    def ToCKSAAGP(self):
        return self._iFeatureEncode("CKSAAGP")

    def ToCKSAAP(self):
        return self._iFeatureEncode("CKSAAP")

    def ToCTDC(self):
        return self._iFeatureEncode("CTDC")

    def ToEAAC(self):
        return self._iFeatureEncode("EAAC")

    def ToDPC(self):
        return self._iFeatureEncode("DPC")

    def ToDDE(self):
        return self._iFeatureEncode("DDE")

    def ToTPC(self):
        return self._iFeatureEncode("TPC")

    def ToGAAC(self):
        return self._iFeatureEncode("GAAC")

    def ToCTDT(self):
        return self._iFeatureEncode("CTDT")

    def ToPWM_p(self):
        X, y = [], []

        db1 = [str(record.seq) for record in SeqIO.parse(self.db1, "fasta")]
        db2 = [str(record.seq) for record in SeqIO.parse(self.db2, "fasta")]
        
        if self.do_PWM_p == False :
            self.PWM_p = feature.MkPWM(db1)
            self.do_PWM_p = True

        NewDBs = dataset.Balance(db1, db2)

        del(db1, db2)
        
        
        for positiveData in feature.GetPWM(NewDBs[0], self.PWM_p):
            X.append(positiveData)
            y.append(0)

        for negativeData in feature.GetPWM(NewDBs[1], self.PWM_p):
            X.append(negativeData)
            y.append(1)
        #exit(self.PWM_p)
        return pd.DataFrame(X), pd.DataFrame(y)
    
    def ToPWM_n(self):
        X, y = [], []

        db1 = [str(record.seq) for record in SeqIO.parse(self.db1, "fasta")]
        db2 = [str(record.seq) for record in SeqIO.parse(self.db2, "fasta")]

        if self.do_PWM_n == False :
            self.PWM_n = feature.MkPWM(db2)
            self.do_PWM_n = True

        NewDBs = dataset.Balance(db1, db2)

        del(db1, db2)
        
        for positiveData in feature.GetPWM(NewDBs[0], self.PWM_n):
            X.append(positiveData)
            y.append(0)

        for negativeData in feature.GetPWM(NewDBs[1], self.PWM_n):
            X.append(negativeData)
            y.append(1)
        
        #exit(self.PWM_n)
        return pd.DataFrame(X), pd.DataFrame(y)
    
    def ToPWM_all(self):
        X, y = [], []

        db1 = [str(record.seq) for record in SeqIO.parse(self.db1, "fasta")]
        db2 = [str(record.seq) for record in SeqIO.parse(self.db2, "fasta")]

        if self.do_PWM_all == False :
            self.PWM_all = feature.MkPWM(db1+db2)
            self.do_PWM_all = True
        

        NewDBs = dataset.Balance(db1, db2)

        del(db1, db2)
        
        for positiveData in feature.GetPWM(NewDBs[0], self.PWM_all):
            X.append(positiveData)
            y.append(0)

        for negativeData in feature.GetPWM(NewDBs[1], self.PWM_all):
            X.append(negativeData)
            y.append(1)
        
        #exit(self.PWM_all)
        return pd.DataFrame(X), pd.DataFrame(y)
    
    def ToPWM_d(self):
        X, y = [], []

        db1 = [str(record.seq) for record in SeqIO.parse(self.db1, "fasta")]
        db2 = [str(record.seq) for record in SeqIO.parse(self.db2, "fasta")]
        
        if self.do_PWM_p == False :
            self.PWM_p = feature.MkPWM(db1)
            self.do_PWM_p = True
        
        if self.do_PWM_n == False :
            self.PWM_n = feature.MkPWM(db2)
            self.do_PWM_n = True
            
        if self.do_PWM_d == False :
            self.PWM_d = feature.MkPWMd(self.PWM_p, self.PWM_n)
            self.do_PWM_d = True
        
        NewDBs = dataset.Balance(db1, db2)

        del(db1, db2)
        
        for positiveData in feature.GetPWM(NewDBs[0], self.PWM_d):
            X.append(positiveData)
            y.append(0)

        for negativeData in feature.GetPWM(NewDBs[1], self.PWM_d):
            X.append(negativeData)
            y.append(1)
        
        #exit(self.PWM_d[20])
        return pd.DataFrame(X), pd.DataFrame(y)
 
    def ToPWM_d2(self):
        X, y = [], []

        db1 = [str(record.seq) for record in SeqIO.parse(self.db1, "fasta")]
        db2 = [str(record.seq) for record in SeqIO.parse(self.db2, "fasta")]

        if self.do_PWM_p == False :
            self.PWM_p = feature.MkPWM(db1)
            self.do_PWM_p = True
        
        if self.do_PWM_n == False :
            self.PWM_n = feature.MkPWM(db2)
            self.do_PWM_n = True
            
        if self.do_PWM_d2 == False :
            self.PWM_d2 = feature.MkPWMd2(self.PWM_p, self.PWM_n)
            self.do_PWM_d2 = True
        
        NewDBs = dataset.Balance(db1, db2)

        del(db1, db2)
        
        for positiveData in feature.GetPWM(NewDBs[0], self.PWM_d2):
            X.append(positiveData)
            y.append(0)

        for negativeData in feature.GetPWM(NewDBs[1], self.PWM_d2):
            X.append(negativeData)
            y.append(1)
        
        #exit(self.PWM_d2)
        return pd.DataFrame(X), pd.DataFrame(y)
   
    def ToPWM_d3(self):
        X, y = [], []

        db1 = [str(record.seq) for record in SeqIO.parse(self.db1, "fasta")]
        db2 = [str(record.seq) for record in SeqIO.parse(self.db2, "fasta")]

        if self.do_PWM_p == False :
            self.PWM_p = feature.MkPWM(db1)
            self.do_PWM_p = True
        
        if self.do_PWM_n == False :
            self.PWM_n = feature.MkPWM(db2)
            self.do_PWM_n = True

        if self.do_PWM_d2 == False :
            self.PWM_d2 = feature.MkPWMd2(self.PWM_p, self.PWM_n)
            self.do_PWM_d2 = True
        
        if self.do_PWM_d3 == False :
            self.PWM_d3 = feature.MkPWMd3(self.PWM_d2)
            self.do_PWM_d3 = True

        NewDBs = dataset.Balance(db1, db2)

        del(db1, db2)
        
        for positiveData in feature.GetPWM(NewDBs[0], self.PWM_d3):
            X.append(positiveData)
            y.append(0)

        for negativeData in feature.GetPWM(NewDBs[1], self.PWM_d3):
            X.append(negativeData)
            y.append(1)
        
        #exit(self.PWM_d3)
        return pd.DataFrame(X), pd.DataFrame(y)
    
    def ToElectric(self):
        X, y = [], []

        db1 = [str(record.seq) for record in SeqIO.parse(self.db1, "fasta")]
        db2 = [str(record.seq) for record in SeqIO.parse(self.db2, "fasta")]

        NewDBs = dataset.Balance(db1, db2)

        del(db1, db2)
        
        for positiveData in feature.Electric(NewDBs[0]):
            X.append(positiveData)
            y.append(0)

        for negativeData in feature.Electric(NewDBs[1]):
            X.append(negativeData)
            y.append(1)

        return pd.DataFrame(X), pd.DataFrame(y)
    
    def ToPolor(self):
        X, y = [], []

        db1 = [str(record.seq) for record in SeqIO.parse(self.db1, "fasta")]
        db2 = [str(record.seq) for record in SeqIO.parse(self.db2, "fasta")]

        NewDBs = dataset.Balance(db1, db2)

        del(db1, db2)
        
        for positiveData in feature.Polor(NewDBs[0]):
            X.append(positiveData)
            y.append(0)

        for negativeData in feature.Polor(NewDBs[1]):
            X.append(negativeData)
            y.append(1)
        
        return pd.DataFrame(X), pd.DataFrame(y)

    def ToAromatic(self):
        X, y = [], []

        db1 = [str(record.seq) for record in SeqIO.parse(self.db1, "fasta")]
        db2 = [str(record.seq) for record in SeqIO.parse(self.db2, "fasta")]

        NewDBs = dataset.Balance(db1, db2)

        del(db1, db2)
        
        for positiveData in feature.Aromatic(NewDBs[0]):
            X.append(positiveData)
            y.append(0)

        for negativeData in feature.Aromatic(NewDBs[1]):
            X.append(negativeData)
            y.append(1)
        
        return pd.DataFrame(X), pd.DataFrame(y)

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
