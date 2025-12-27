from setuptools import setup, find_packages

setup(
    name='cdft_solver',               # Package name
    version='0.2.0',                   # Version number
    author='Your Name',                # Your name
    author_email='your.email@example.com',  
    description='A package for CDFT calculations',
    packages=find_packages(),          # Automatically finds subpackages
    install_requires=[                 # Dependencies your package needs
        'numpy',
        'scipy',
        'matplotlib',
    ],
    python_requires='>=3.8',           # Minimum Python version
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)

