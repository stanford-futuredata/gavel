from setuptools import setup, find_packages

import recoder


setup(
  name='recsys-recoder',
  version=recoder.__version__,
  install_requires=['torch>=0.4.1', 'annoy',
                    'numpy', 'scipy>=1.2.0',
                    'tqdm', 'glog'],
  packages=find_packages(),
  author='Abdallah Moussawi',
  author_email='abdallah.moussawi@gmail.com',
  url='https://github.com/amoussawi/recoder',
  license='MIT'
)
