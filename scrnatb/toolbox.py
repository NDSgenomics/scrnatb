import numpy as np
import pandas as pd
import seaborn as sns
sns.set_color_codes()

from glob import iglob

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


def read_kallisto(sample_path):
    quant_file = sample_path + '/abundance.tsv'
    df = pd.read_table(quant_file, engine='c',
                                   usecols=['target_id', 'tpm'],
                                   index_col=0,
                                   dtype={'target_id': np.str, 'tpm': np.float64})
    return df


def read_kallisto_dir(pattern):
    TPMs = pd.DataFrame()
    for sample_path in tqdm(iglob(pattern)):
        sample_df = read_kallisto(sample_path)
        TPMs[sample_path] = sample_df['tpm']
        
    return TPMs

def read_sailfish(sample_path, isoforms=False):
    if isoforms:
        quant_file = sample_path + '/quant.sf'
    else:
        quant_file = sample_path + '/quant.genes.sf'
    
    df = pd.read_table(quant_file, engine='c',
                                   usecols=['Name', 'TPM'],
                                   index_col=0,
                                   dtype={'Name': np.str, 'TPM': np.float64})
    return df

def read_sailfish_dir(pattern, isoforms=False):
    TPMs = pd.DataFrame()
    for sample_path in tqdm(iglob(pattern)):
        sample_df = read_sailfish(sample_path, isoforms=isoforms)
        TPMs[sample_path] = sample_df['TPM']
        
    return TPMs


def read_cufflinks(sample_path):
    quant_file = sample_path + '/genes.fpkm_tracking'
    df = pd.read_table(quant_file, engine='c',
                                   usecols=['tracking_id', 'FPKM'],
                                   index_col=0,
                                   dtype={'tracking_id': np.str, 'FPKM': np.float64})
    
    df['tracking_id'] = df.index
    df = df.groupby('tracking_id').sum()
    df['TPM'] = df['FPKM'] / df['FPKM'].sum() * 1e6
    
    return df


def read_cufflinks_dir(pattern):
    TPMs = pd.DataFrame()
    for sample_path in tqdm(iglob(pattern)):
        sample_df = read_cufflinks(sample_path)
        TPMs[sample_path] = sample_df['TPM']
        
    return TPMs


def read_quants(sample_dir, suffix='_salmon_out', tool='salmon', isoforms=False):
    pattern = '{}*{}'.format(sample_dir, suffix)
    
    if tool == 'kallisto':
        return read_kallisto_dir(pattern)
    
    elif tool == 'sailfish':
        return read_sailfish_dir(pattern)
    
    elif tool == 'salmon':
        return read_sailfish_dir(pattern, isoforms=isoforms)
    
    elif tool == 'cufflinks':
        return read_cufflinks_dir(pattern)
