#!/bin/bash
# Execution script for extracting experiences using LLM as a Judge

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
SWE_LITE_TEST_DIR="${PROJECT_ROOT}/swe_lite_test_gpt_5_mini"
# export ANTHROPIC_API_KEY=your-api-key-here
# export OPENAI_API_KEY=your-api-key-here
# Please set API key from environment variables, or uncomment and fill in your key


# Default configuration
MODEL_NAME="${MODEL_NAME:-anthropic/claude-sonnet-4-20250514}"
OUTPUT_FILE="${OUTPUT_FILE:-${SWE_LITE_TEST_DIR}/extracted_experiences_llm_gpt_5_mini.jsonl}"
FILTER_IDS_FILE="${FILTER_IDS_FILE:-}"

# Check default filter file
if [ -z "$FILTER_IDS_FILE" ] && [ -f "${SCRIPT_DIR}/exp_ids.txt" ]; then
    FILTER_IDS_FILE="${SCRIPT_DIR}/exp_ids.txt"
fi

# Check if directory exists
if [ ! -d "$SWE_LITE_TEST_DIR" ]; then
    echo "Error: Directory $SWE_LITE_TEST_DIR does not exist"
    exit 1
fi

# Check if cleaned files exist
CLEANED_FILES=$(find "$SWE_LITE_TEST_DIR" -name "*_cleaned.json" | wc -l)
if [ "$CLEANED_FILES" -eq 0 ]; then
    echo "Error: No cleaned trajectory files (*_cleaned.json) found in $SWE_LITE_TEST_DIR"
    echo "Please run the cleaning script first"
    exit 1
fi

echo "=========================================="
echo "LLM as a Judge Experience Extraction"
echo "=========================================="
echo "Directory: $SWE_LITE_TEST_DIR"
echo "Model: $MODEL_NAME"
echo "Output: $OUTPUT_FILE"
if [ -n "$FILTER_IDS_FILE" ]; then
    echo "Filter IDs file: $FILTER_IDS_FILE"
fi
echo "Found $CLEANED_FILES cleaned trajectory files"
echo "=========================================="
echo ""

# Run extraction script
cd "$PROJECT_ROOT"
CMD="python -m minisweagent.experience.extract_with_llm_judge \
    \"$SWE_LITE_TEST_DIR\" \
    --model \"$MODEL_NAME\" \
    --output \"$OUTPUT_FILE\" \
    --pattern \"*_cleaned.json\""

if [ -n "$FILTER_IDS_FILE" ]; then
    CMD="$CMD --filter-ids \"$FILTER_IDS_FILE\""
fi

eval $CMD

echo ""
echo "=========================================="
echo "Extraction complete!"
echo "Results saved to: $OUTPUT_FILE"
echo "=========================================="

