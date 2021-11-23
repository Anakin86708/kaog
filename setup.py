from setuptools import setup

with open("requirements.txt", "r") as fh:
   requirements = fh.readlines()

setup(
   name='kaog',
   version='0.2',
   description='A module implementing KAOG',
   author='Ariel Tadeu da Silva',
   author_email='silva.ariel@icloud.com',
   packages=['kaog'],
   install_requires=[req.strip() for req in requirements if req[:2] != "# "],
)
