from setuptools import setup, find_packages

setup(
    name='dr',  # Required
    version='0.0.0',  # Required

    packages=find_packages(exclude=[]),  # Required

    python_requires='>=3.6',

    install_requires=[
        'gym @ git+https://github.com/DartEnv/dart-env.git'
    ],
    extras_require={  # Optional
        'dart': ['pydart2'],
        'mujoco': ['mujoco-py'],
        'all': ['pydart2', 'mujoco-py'],
    },

)
