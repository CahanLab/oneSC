import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="OneCC_bool", # Replace with your own username
    version="0.0.1",
    author="Dan Peng",
    author_email="dpeng5@jhu.edu",
    description="This is a prototype for OneCC Boolean version",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
