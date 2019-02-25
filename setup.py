import os
from setuptools import setup, find_packages

def read_requirements():
    """Parse requirements from requirements.txt."""
    reqs_path = os.path.join('.', 'requirements.txt')
    with open(reqs_path, 'r') as f:
        requirements = [line.rstrip() for line in f]
        return requirements

print(read_requirements())
setup(
    name='dr',  # Required
    version='0.0.0',  # Required

    packages=find_packages(exclude=[]),  # Required

    python_requires='>=3.6',

    install_requires=read_requirements(),
    extras_require={
        'dart': ['pydart2'],
        'mujoco': ['mujoco-py'],
        'all': ['pydart2', 'mujoco-py'],
    },

)
