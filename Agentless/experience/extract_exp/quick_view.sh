#!/bin/bash
# Quick view of Experience statistics

# Set working directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default file path
EXPERIENCE_FILE="/home3/yaoqi/Agentless-Exp/experience/extracted_experiences.jsonl"

# Check if file exists
if [ ! -f "$EXPERIENCE_FILE" ]; then
    echo "Error: Experience file not found: $EXPERIENCE_FILE"
    echo "Please run ./run_extraction.sh first to extract Experience"
    exit 1
fi

# Display menu
echo "=========================================="
echo "Experience Viewer"
echo "=========================================="
echo ""
echo "Select an action:"
echo "1. View statistics"
echo "2. View stage distribution"
echo "3. View confidence distribution"
echo "4. View samples (first 5)"
echo "5. List all Issues"
echo "6. Search for specific Issue"
echo "7. Export to JSON"
echo "8. View all information"
echo "0. Exit"
echo ""

read -p "Enter option [0-8]: " choice

case $choice in
    1)
        python view_experiences.py --action stats
        ;;
    2)
        python view_experiences.py --action stages
        ;;
    3)
        python view_experiences.py --action confidence
        ;;
    4)
        python view_experiences.py --action samples --n 5
        ;;
    5)
        python view_experiences.py --action list
        ;;
    6)
        read -p "Enter Issue ID: " issue_id
        python view_experiences.py --action search --issue-id "$issue_id"
        ;;
    7)
        read -p "Enter output file path: " output_file
        python view_experiences.py --action export --output "$output_file"
        ;;
    8)
        echo ""
        echo "=== Statistics ==="
        python view_experiences.py --action stats
        echo ""
        echo "=== Stage Distribution ==="
        python view_experiences.py --action stages
        echo ""
        echo "=== Confidence Distribution ==="
        python view_experiences.py --action confidence
        echo ""
        echo "=== Sample Display ==="
        python view_experiences.py --action samples --n 3
        ;;
    0)
        echo "Exit"
        exit 0
        ;;
    *)
        echo "Invalid option"
        exit 1
        ;;
esac
