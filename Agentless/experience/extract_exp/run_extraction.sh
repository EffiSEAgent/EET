#!/bin/bash
# Run Experience extraction script

# Set working directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Set PYTHONPATH to include the parent directory (Agentless-Exp)
# This allows importing the experience module
export PYTHONPATH="$(dirname "$(dirname "$SCRIPT_DIR")"):${PYTHONPATH}"

echo "=========================================="
echo "Experience Extraction Tool"
echo "=========================================="
echo ""

# Check Python environment
if ! command -v python &> /dev/null; then
    echo "Error: Python not found"
    exit 1
fi

echo "Python version:"
python --version
echo ""

# Run extraction script
echo "Starting Experience extraction..."
echo ""
python extract_from_verified.py

# Check execution result
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Extraction completed!"
    echo "=========================================="
    echo ""
    
    # Display statistics
    echo "Viewing statistics..."
    python view_experiences.py --action stats
    
    echo ""
    echo "Viewing stage distribution..."
    python view_experiences.py --action stages
    
    echo ""
    echo "Viewing confidence distribution..."
    python view_experiences.py --action confidence
else
    echo ""
    echo "=========================================="
    echo "Extraction failed, please check errors"
    echo "=========================================="
    exit 1
fi
