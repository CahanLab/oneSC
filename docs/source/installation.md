### Installation
We recommend creating a new conda environment (with python version >= 3.9) and install OneSC. Open terminal and type the following code. 
```
# type this into the terminal 
conda create -n OneSC_run python=3.9
conda activate OneSC_run 
```
In the conda environment, you can install package via
```
# should be able to install onesc and the necessary dependencies. 
pip install git+https://github.com/CahanLab/oneSC.git
```

**Note - Additional Installations**

To use `pip install git` you may need to install git first.
```
conda install anaconda::git
```
You may also need to install cairo before installing OneSC. 

**Ubuntu**
```
sudo apt-get install libcairo2-dev
```
**Mac**
```
brew install cairo
```