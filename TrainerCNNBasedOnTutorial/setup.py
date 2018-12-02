from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
  'pillow==4.0.0', 'tensorflow==1.12', 'numpy==1.15.4', 'Keras==2.2.0'
]

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    requires=[]
)
