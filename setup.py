from setuptools import setup, find_packages

print(find_packages())
setup(name='deeprecsys',
      version='0.01',
      install_requires=[
            'pandas>=1.2.1',
            'torch>=1.7.0',
            'tqdm',
            'numpy',
            'scipy',
            'scikit_learn',
            'tensorboard'
      ],
      packages=['deeprecsys'],
      package_data={"deeprecsys": ["py.typed"]})