# export OPENAI_API_KEY=your-api-key-here
# Please set API key from environment variables, or uncomment and fill in your key

python3 extract_with_llm_judge.py merged_contexts.jsonl \
    --workers 8 \
    --model openai/gpt-5 \
    --output extracted_experiences_llm.jsonl