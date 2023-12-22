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

def Text2Vector(rawSSEPSSM):
    Features = []
    Residues = rawSSEPSSM.strip().split("\n")

    if len(Residues) <= 10:
        return []

    for res in Residues[2:]:
        Features.append([float(x) for x in res.split("\t")[2:]])

    return Features
