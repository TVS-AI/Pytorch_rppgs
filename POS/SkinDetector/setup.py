#!/usr/bin/env python
# -*- coding: utf-8 -*-

import ez_setup
ez_setup.use_setuptools()

from setuptools import setup
setup(
    name="Skin Detector",
    version="1.0a Prototype",
    url='',
    author='Will Brennan',
    author_email='william.brennan@skytales.com',
    license='GPL',
    install_requires=["numpy"], )

from setuptools import setup, find_packages

with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()
setup(
    name='Skin Detector',
    version='prototype',
    description='A speedy skin-detector based upon colour thresholding',
    long_description=readme,
    author='WillBrennan',
    author_email='WillBrennan@users.noreply.github.com',
    url='https://github.com/WillBrennan/SkinDetector',
    license=license,
    install_requires=required,
    packages=find_packages(exclude=('tests', 'docs')))
