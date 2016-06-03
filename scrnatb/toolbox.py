import pandas as pd
import seaborn as sns
sns.set_color_codes()

from tqdm import tqdm

def exogen_scale_tpm(tpm, prefix='ERCC-'):
    ''' Takes an expression DataFrame with either TPM or FPKM,
    and returns a DataFrame where genes starting with prefix
    have been removed and TPM values rescaled.
    '''
    exogen_idx = filter(lambda s: prefix in s, tpm.index)
    etpm = tpm.drop(exogen_idx)
    etpm = etpm / etpm.sum() * 1e6

    return etpm
