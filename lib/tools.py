##### Import
import hashlib



##### Functions
def GetSHA256(input_string):
    input_bytes = input_string.encode('utf-8')

    return hashlib.sha256(input_bytes).hexdigest()
