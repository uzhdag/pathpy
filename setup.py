#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ["numpy", "scipy"]

setup_requirements = ['pytest-runner']

test_requirements = ['pytest']

setup(
    author="Ingo Scholtes",
    author_email='ischoltes@ethz.ch',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    description="A python package for the analysis of sequential data on pathways and temporal networks from the perspective of higher-order network models.",
    install_requires=requirements,
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='pathpy',
    name='pathpy',
    packages=find_packages(include=['pathpy']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/IngoScholtes/pathpy',
    version='1.2.1',
    zip_safe=False,
    data_files=[('.', ['DESCRIPTION.rst', 'README.rst', 'HISTORY.rst'])]
)
