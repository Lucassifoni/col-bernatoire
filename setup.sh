#!/bin/bash

# Exit on error
set -e

# Create a virtual environment
echo "Creating virtual environment..."
python3.11 -m venv venv

# Activate the virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install system dependencies if possible (for PDAL and LAZ support)
if command -v apt-get &> /dev/null; then
    echo "Detecting apt package manager, installing system dependencies..."
    sudo apt-get update
    sudo apt-get install -y libpdal-dev pdal liblas-dev liblapack-dev
elif command -v brew &> /dev/null; then
    echo "Detecting Homebrew, installing system dependencies..."
    brew install pdal laszip
elif command -v yum &> /dev/null; then
    echo "Detecting yum package manager, installing system dependencies..."
    sudo yum install -y pdal-devel laszip-devel
else
    echo "Could not detect package manager. Please install PDAL and laszip manually if needed."
fi

# Install dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt

# Make the Python script executable
chmod +x laz_to_stl.py

echo "Setup complete! To activate the environment, run:"
echo "source venv/bin/activate"
echo ""
echo "To convert a LAZ file to STL, run:"
echo "./laz_to_stl.py input_file.laz"
echo ""
echo "Note: For COPC files, use:"
echo "./laz_to_stl.py your_file.copc.laz"
