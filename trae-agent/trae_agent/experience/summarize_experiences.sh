#!/bin/bash

# Set API key if needed
# export OPENAI_API_KEY=your-api-key-here
# Please set API key from environment variables, or uncomment and fill in your key

# Run summarization script
python3 summarize_experiences.py extracted_experiences_llm.jsonl \
    --model openai/gpt-5-mini \
    --output extracted_experiences_summarized.jsonl

