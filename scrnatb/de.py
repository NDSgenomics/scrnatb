import numpy as np
import pandas as pd

import statsmodels.formula.api as smf
from statsmodels.sandbox.stats.multicomp import multipletests

from tqdm import tqdm

def lr_tests(sample_info, expression_matrix, full_model, reduced_model='expression ~ 1'):
    tmp = sample_info.copy()

    fit_results = pd.DataFrame(index=expression_matrix.index)

    gene = expression_matrix.index[0]
    tmp['expression'] = expression_matrix.ix[gene]
    m1 = smf.ols(full_model, tmp).fit()
    m2 = smf.ols(reduced_model, tmp).fit()

    for param in m1.params.index:
        fit_results['full ' + param] = np.nan

    params = m1.params.add_prefix('full ')
    fit_results.ix[gene, params.index] = params

    for param in m2.params.index:
        fit_results['reduced ' + param] = np.nan

    params = m2.params.add_prefix('reduced ')
    fit_results.ix[gene, params.index] = params

    fit_results['pval'] = np.nan

    fit_results.ix[gene, 'pval'] = m1.compare_lr_test(m2)[1]

    for gene in tqdm(expression_matrix.index[1:]):
        tmp['expression'] = expression_matrix.ix[gene]

        m1 = smf.ols(full_model, tmp).fit()
        params = m1.params.add_prefix('full ')
        fit_results.ix[gene, params.index] = params

        m2 = smf.ols(reduced_model, tmp).fit()
        params = m2.params.add_prefix('reduced ')
        fit_results.ix[gene, params.index] = params

        fit_results.ix[gene, 'pval'] = m1.compare_lr_test(m2)[1]

    fit_results['qval'] = multipletests(fit_results['pval'], method='b')[1]
    
    return fit_results
