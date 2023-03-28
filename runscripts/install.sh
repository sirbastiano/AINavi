#!/bin/bash

# Check if Python is installed
if ! command -v python &> /dev/null
then
    echo "Python is not installed. Please install Python and try again."
    exit 1
fi

# Check if Python is the correct version
if ! python -c "import sys; sys.exit(0 if sys.version_info >= (3, 6) else 1)"; then
    echo "Python is not the correct version. Please install Python 3.6 or higher and try again."
    exit 1
fi

# Check if Conda is installed
if ! command -v conda &> /dev/null
then
    echo "Conda is not installed. Please install Conda and try again."
    echo "See https://docs.anaconda.com/anaconda/install/ for more information."
    
    exit 1
fi

# Create a new Conda environment called AINavi
conda create -n AINavi python=3.8 -y

# Activate the AINavi environment
source $(conda info --base)/etc/profile.d/conda.sh
if conda activate AINavi; then
    echo "AINavi environment activated"
else
    echo "AINavi environment not found"
    exit 1
fi

# Run the setup.py script
python setup.py develop
