from setuptools import find_packages,setup
from typing import List

hypen_dot = '-e .'
requirements=[]
def get_requirements(file_path:str) ->List[str]:
    with open(file_path,'r') as f:
        requirements=f.readlines()
        requirements=[req.replace('\n','')for req in requirements]

        if hypen_dot in requirements:
            requirements.remove(hypen_dot)
setup(
    name='mlproject',
    version='0.0.1',
    author='dave_mahn',
    author_email='ikeu4511@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirement.txt'))