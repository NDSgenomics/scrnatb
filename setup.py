#!/usr/bin/env python

from setuptools import setup

setup(
    name='scrnatb',
    version='0.0',
    packages=['scrnatb'],
    install_requires=[
        'GPy',
        'seaborn',
        'scipy',
        'tqdm',
        'matplotlib',
        'statsmodels',
        'patsy',
        'GPclust'
    ],
    dependency_links=['https://github.com/SheffieldML/GPclust/tarball/master#egg=GPclust-0.1.0']
)
