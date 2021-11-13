from setuptools import setup

setup(
   name='kaog',
   version='0.2',
   description='A module implementing KAOG',
   author='Ariel Tadeu da Silva',
   author_email='silva.ariel@icloud.com',
   packages=['kaog'],
   install_requires=['networkx', 'numpy', 'pandas', 'sklearn', 'matplotlib', 'distython'],
   # external packages as dependencies
)
