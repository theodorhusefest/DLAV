from setuptools import find_packages, setup

REQUIRED_PACKAGES = ['matplotlib', 'scikit-image', 'pytorch', 'opencv-python']

setup(
    name='src',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Setup for DLAV'
)