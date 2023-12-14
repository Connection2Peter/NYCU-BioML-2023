##### Import
import subprocess



##### Functions
def GenerateBySeq(prg, seq, tdb, rdb, search, algo, matrix, threads, inifile):
    try:
        RET = subprocess.check_output([
                prg, "-seq", str(seq), "-tdb", tdb,
                "-rdb", rdb, "-search", search, "-algo", algo,
                "-matrix", matrix, "-threads", str(threads), "-inifile", inifile,
            ], stderr=subprocess.DEVNULL
        ).decode('utf-8')
    except:
        RET = ""

    return RET

def GenerateBySeqAndSave(prg, seq, tdb, rdb, search, algo, matrix, threads, inifile, saveTo):
    try:
        RET = subprocess.check_output([
                prg, "-seq", str(seq), "-tdb", tdb, "-rdb", rdb, 
                "-search", search, "-algo", algo, "-matrix", matrix, 
                "-threads", str(threads), "-inifile", inifile, "-outfile", saveTo
            ], stderr=subprocess.DEVNULL
        ).decode('utf-8')
    except:
        RET = ""

    return RET

def Parser(rawSSEPSSM):
    Features = []
    SeqPSSMs = rawSSEPSSM.split("## SSE-PSSM:")[1:]

    for seqPSSM in SeqPSSMs:
        Columns = []
        Residue = seqPSSM.strip().split("\n")

        if len(Residue) <= 10:
            continue

        for res in Residue:
            if len(res) > 1000:
                Columns.append([float(i) for i in res.strip('\t').split("\t")[2:]])

        Features.append(Columns)

    return Features
