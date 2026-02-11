#!/bin/bash
set -e

# Test uv installation instructions
echo "Testing uv installation steps..."

# Check if uv is installed, if not try to install it
if ! command -v uv &> /dev/null; then
    echo "uv not found, installing..."
    pip install uv
fi

# Create a clean test directory
TEST_DIR=$(mktemp -d)
echo "Working in $TEST_DIR"

# 1. Create venv
echo "Creating venv..."
uv venv "$TEST_DIR/GO" --python 3.12 # Using 3.12 as 3.13 might not be available in all envs, aligning with Dockerfile which uses vLLM base (likely 3.10-3.12)
source "$TEST_DIR/GO/bin/activate"

# 2. Install dependencies
echo "Installing dependencies..."
# We need to be in the root of the repo to check requirements.txt
# Assuming this script is run from repo root or we know where it is.
# Using current directory assuming script is run from repo root.
if [ -f "requirements.txt" ]; then
    uv pip install -r requirements.txt
else
    echo "requirements.txt not found!"
    exit 1
fi

# 3. Install package
echo "Installing GenomeOcean..."
uv pip install -e ".[all]"

# 4. Verify import
echo "Verifying import..."
python -c "import genomeocean; print('GenomeOcean imported successfully')"
python -c "import vllm; print('vllm imported successfully')"

echo "Test passed!"
rm -rf "$TEST_DIR"
