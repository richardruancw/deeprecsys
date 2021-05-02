from setuptools import setup, find_packages

print(find_packages())
setup(name='deeprecsys',
      version='0.01',
      packages=['deeprecsys'],
      package_data={"deeprecsys": ["py.typed"]})