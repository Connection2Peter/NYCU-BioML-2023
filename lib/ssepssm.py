##### Import
import re, subprocess



##### Functions
def isValidSeq(input_str):
    pattern = re.compile("^[ACDEFGHIKLMNPQRSTVWY]+$")

    return bool(pattern.match(input_str))

def getSSEPSSM(ip, seq):
    url = f"http://{ip}:12387/ssepssm?seq={seq}"

    try:
        response = requests.get(url)
        print(response.text)
    except requests.exceptions.HTTPError as errh:
        print(f"HTTP Error: {errh}")
    except requests.exceptions.ConnectionError as errc:
        print(f"Error Connecting: {errc}")
    except requests.exceptions.Timeout as errt:
        print(f"Timeout Error: {errt}")
    except requests.exceptions.RequestException as err:
        print(f"Request Exception: {err}")
    
    return response

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
