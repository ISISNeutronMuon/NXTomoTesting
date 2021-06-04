from setuptools import setup, find_packages
from nxtomowriter import version

setup(
    name='nxtomowriter',
    version=version,
    description='Output tomography data into a nexus file using the NxTomo format',
    packages=find_packages(include=['nxtomowriter']),
    install_requires=[
        'h5py>=2.10.0',
        'numpy>=1.15.4',
        'tifffile>=2020.6.3',
        'tqdm>=4.40'
    ]
)