import os
import tempfile
import pandas as pd
from Bio import SeqIO
from lib import dataset


class IFeature:

    def __init__(self, db1, db2, encoder):
        self.db1     = db1
        self.db2     = db2
        self.temp    = ""
        self.encoder = encoder
        self.min_len = 0


    def iFeatureEncode(self):
        db1 = [str(record.seq) for record in SeqIO.parse(self.db1, "fasta")]
        db2 = [str(record.seq) for record in SeqIO.parse(self.db2, "fasta")]

        NewDBs = dataset.Balance(db1, db2)

        total_fasta = ""
        for i, seq in enumerate(NewDBs[0]):
            total_fasta += f">Pos{i}\n" + seq + "\n"
        
        for i, seq in enumerate(NewDBs[1]):
            total_fasta += f">Neg{i}\n" + seq + "\n"

        self.min_len = len(NewDBs[0])
        
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

        temp_out = temp_fastaFile[:-12] + "_" + self.encoder + ".csv"

        cmd = f"python bin/iFeature/iFeature.py --file {temp_fastaFile} --type {self.encoder} --out {temp_out} > /dev/null"

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

        self.temp = temp_out

        return self.parse_csv()


    def parse_csv(self):
        X, y = [], []
        try:
            content = pd.read_csv(self.temp, sep='\t')
        except:
            print("Error: Cannot open file " + self.temp)
            return

        for i in range (self.min_len * 2):
            X.append( list(content.iloc[i, 1:]) )

        y = [0] * self.min_len + [1] * self.min_len

        try:
            os.remove(self.temp)
        except:
            print("Error: Cannot remove file " + self.temp)
            return

        return pd.DataFrame(X), pd.DataFrame(y)
