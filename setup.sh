#!/bin/bash

# Download anaconda installer
wget https://repo.anaconda.com/archive/Anaconda3-2023.07-2-Linux-x86_64.sh

# Install anaconda
shasum -a 256 Anaconda3*.sh
bash Anaconda3*.sh

echo "Installed Anaconda; run 'source ~/.bashrc; conda env create -f environment.yml' to complete setup."
echo "Run setup_tf.sh to complete tensorflow 2 setup."
echo "Make sure to update git config --global user.name and user.email."

# Delete install scripts
rm Anaconda3*.sh