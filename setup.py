#!/usr/bin/env python
"""
Create easy-to-use Query objects that can apply on
NumPy structured arrays, astropy Table, and Pandas DataFrame.
Project website: https://github.com/yymao/easyquery
The MIT License (MIT)
Copyright (c) 2017 Yao-Yuan Mao (yymao)
http://opensource.org/licenses/MIT
"""

from setuptools import setup

setup(
    name='easyquery',
    version='0.1.2',
    description='Create easy-to-use Query objects that can apply on NumPy structured arrays, astropy Table, and Pandas DataFrame.',
    url='https://github.com/yymao/easyquery',
    download_url = 'https://github.com/yymao/easyquery/archive/v0.1.2.zip',
    author='Yao-Yuan Mao',
    author_email='yymao.astro@gmail.com',
    maintainer='Yao-Yuan Mao',
    maintainer_email='yymao.astro@gmail.com',
    license='MIT',
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
    ],
    keywords='easyquery query numpy',
    py_modules=['easyquery'],
    install_requires=['numpy', 'numexpr'],
)
