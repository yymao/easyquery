#!/usr/bin/env python
"""
Create easy-to-use Query objects that can apply on
NumPy structured arrays, astropy Table, and Pandas DataFrame.
Project website: https://github.com/yymao/easyquery
The MIT License (MIT)
Copyright (c) 2017-2021 Yao-Yuan Mao (yymao)
http://opensource.org/licenses/MIT
"""

import os
from setuptools import setup

_name = 'easyquery'
_version = None
with open(os.path.join(os.path.dirname(__file__), '{}.py'.format(_name))) as _f:
    for _l in _f:
        if _l.startswith('__version__ = '):
            _version = _l.partition('=')[2].strip().strip('\'').strip('"')
            break
if not _version:
    raise ValueError('__version__ not define!')

setup(
    name=_name,
    version=_version,
    description='Create easy-to-use Query objects that can apply on NumPy structured arrays, astropy Table, and Pandas DataFrame.',
    url='https://github.com/yymao/{}'.format(_name),
    download_url='https://github.com/yymao/{}/archive/v{}.tar.gz'.format(_name, _version),
    author='Yao-Yuan Mao',
    author_email='yymao.astro@gmail.com',
    maintainer='Yao-Yuan Mao',
    maintainer_email='yymao.astro@gmail.com',
    license='MIT',
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    keywords='easyquery query numpy',
    py_modules=[_name],
    python_requires='>=3.6',
    install_requires=['numpy>=1.7', 'numexpr>=2.0'],
)
