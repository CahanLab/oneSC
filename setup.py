import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="onesc", 
    version="0.1.1",
    author="Dan Peng",
    author_email="dpeng5@jhu.edu",
    description="This is a prototype for OneSC",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'click',
        'pygad>=3.1.0',
        'pygam>=0.8.0',
        'numpy>=1.21.5, <1.28.0', 
        'pandas>=1.4.3',
        'statsmodels>=0.13.2', 
        'scipy>=1.8.0, <1.12.0', 
        'scanpy>=1.9.1', 
        'joblib>=1.1.0',
        'networkx>=2.8.8, <=3.2.1',
        'adjustText>=1.1.1', 
        'igraph>=0.11.5'
    ],
    python_requires='>=3.9',
)
