#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages
from pathpy import __version__


with open('README.rst', encoding='utf-8') as readme_file, open('HISTORY.rst', encoding='utf-8') as history_file:
    readme = readme_file.read()
    history = history_file.read()

install_requirements = ['numpy', 'scipy']

setup_requirements = ['pytest-runner']

setup(
    author="Ingo Scholtes",
    author_email='scholtes@ifi.uzh.ch',
    license='AGPL-3.0+',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    description="An OpenSource python package for the analysis and visualisation of time series data on"
                "complex networks with higher- and multi-order graphical models.",
    install_requires=install_requirements,
    setup_requires=setup_requirements,
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    python_requires='>=3.5',
    keywords='network analysis temporal networks pathways sequence modeling graph mining',
    name='pathpy',
    packages=find_packages(),
    test_suite='tests',
    url='https://www.pathpy.net',
    version=__version__,
    zip_safe=False
)
