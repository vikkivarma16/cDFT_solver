from setuptools import setup, find_packages

setup(
    name='cdft_solver',               # Package name
    version='0.2.0',                   # Version number
    author='vikkivarma16',                # Your name
    author_email='vikkivarma16@gmail.com', 
    url='https://github.com/vikkivarma16/cDFT_solver',
    description="A package for classical density functional and integral equation theory based calculation.",
    packages=find_packages(),          # Automatically finds subpackages
    include_package_data=True,
    package_data={
        "cdft_solver": ["*.so"],   # include all .so files
    },
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

