#!/usr/bin/env python3
"""
Extract experience records from merged_contexts.jsonl using LLM as a Judge
Adapted for trae-agent tool format (bash, sequentialthinking, str_replace_based_edit_tool, etc.)
"""
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import re
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from experience.models import Experience
    USE_EXPERIENCE_MODEL = True
except ImportError:
    print("Error: Cannot import Experience model")
    sys.exit(1)

try:
    import litellm  # type: ignore
    USE_LITELLM = True
except ImportError:
    print("Error: Please install litellm: pip install litellm")
    sys.exit(1)


JUDGE_PROMPT_TEMPLATE = """You are an expert judge tasked with extracting valuable experiences from agent interaction dialogues. Your goal is to identify meaningful steps that contribute to solving programming tasks and extract structured experience records.

## Task
Analyze the conversation history and extract experience records for each significant step where the assistant interacts with the system. Focus on steps that:
1. Show clear reasoning and planning (sequential_thinking tool calls)
2. Execute meaningful commands (bash, str_replace_based_edit_tool, json_edit_tool)
3. Produce useful outputs or insights
4. Contribute to solving the issue

## Available Tools
The agent uses these tools:
- **bash**: Execute shell commands
- **sequentialthinking**: Structured problem-solving with sequential thoughts
- **str_replace_based_edit_tool**: File operations (view, create, str_replace, insert)
- **json_edit_tool**: JSON file editing using JSONPath
- **task_done**: Signal task completion

## Input Format
You will receive:
- issue_id: The identifier of the issue being solved
- issue_description: Brief description of the problem
- contexts: A list of conversation messages with roles: system, user, tool_call, tool_result

## Output Format
For each significant step, extract an experience record in JSON format. Return a JSON array of experience objects:

```json
[
  {{
    "task_summary": "Brief summary of what this step accomplishes (max 200 chars)",
    "confidence": 85,
    "output": "Relevant output from the tool execution (max 1000 chars, truncate if needed)",
    "reasoning": "Why this step is valuable and what makes it significant"
  }}
]
```

## Evaluation Criteria
1. **Task Summary**: Create a concise, descriptive summary (format: "Task Type: Brief description")
   - Task types: "Find Files", "Read Code", "Analyze Logic", "Modify Code", "Run Tests", "Debug Issue", "Sequential Thinking", etc.
   - Description should capture the key intent and action
   - Include tool name if relevant (e.g., "Sequential Thinking: Plan approach to locate bug")

2. **Confidence Score** (1-100): Assess based on:
   - Tool success: If tool_result shows success, increase confidence
   - Output quality: If output provides useful information, increase confidence
   - Step significance: If step directly contributes to solution, increase confidence
   - Base confidence: 50-60 for exploratory steps, 70-85 for successful actions, 85-95 for final solution steps

3. **Output**: Extract relevant output from the tool_result message
   - Focus on useful information, not error messages or empty outputs
   - For sequential_thinking: Include key insights from the thought
   - For bash: Include command output (truncate if too long)
   - For file operations: Include relevant file content or operation result
   - Truncate to 1000 characters if too long

4. **Reasoning**: Explain why this step is valuable (optional but helpful)

## Filtering Guidelines
- Skip trivial steps (e.g., simple ls without clear purpose)
- Skip steps with empty or unhelpful outputs
- Focus on steps that show problem-solving progression
- Prioritize steps that lead to understanding or fixing the issue
- Include sequential_thinking steps that show clear reasoning
- Don't include more than 8 steps, only include the most important ones.

## Example
Input:
- issue_id: "astropy__astropy-6938"
- issue_description: "Possible bug in io.fits related to D exponents..."
- contexts: [system message, user prompt, tool_call (sequentialthinking), tool_result, tool_call (bash), tool_result...]

Output:
```json
[
  {{
    "task_summary": "Sequential Thinking: Plan approach to locate bug in fitsrec.py",
    "confidence": 75,
    "output": "Start by exploring repository to find files related to fitsrec.py and io.fits...",
    "reasoning": "This step establishes the problem-solving strategy, which is crucial for systematic debugging"
  }},
  {{
    "task_summary": "Find Files: Locate fitsrec.py using grep",
    "confidence": 85,
    "output": "/testbed/astropy/io/fits/fitsrec.py:1262:        # Replace exponent separator...",
    "reasoning": "Successfully located the problematic code, which is essential for fixing the bug"
  }}
]
```

Now analyze the following conversation and extract experience records:

Issue ID: {issue_id}
Issue Description: {issue_description}

Contexts:
{conversation_text}

Extract all valuable experience records as a JSON array:"""


def format_conversation_for_judge(contexts: List[Dict]) -> str:
    """Format message list into text suitable for LLM reading"""
    formatted_lines = []
    
    for i, ctx in enumerate(contexts):
        role = ctx.get("role", "unknown")
        content = ctx.get("content", "")
        
        if len(content) > 5000:
            content = content[:5000] + "\n... [truncated]"
        
        formatted_lines.append(f"[{role.upper()} {i+1}]")
        formatted_lines.append(content)
        formatted_lines.append("")
    
    return "\n".join(formatted_lines)


def extract_tool_info_from_contexts(contexts: List[Dict], tool_call_idx: int) -> Dict[str, Any]:
    """Extract tool call and result information from contexts"""
    tool_info = {
        "tool_name": None,
        "tool_arguments": None,
        "tool_result": None,
        "step_index": tool_call_idx
    }
    
    if tool_call_idx < len(contexts):
        tool_call = contexts[tool_call_idx]
        if tool_call.get("role") == "tool_call":
            content = tool_call.get("content", "")
            tool_match = re.search(r'Tool Call:\s*(\w+)', content)
            if tool_match:
                tool_info["tool_name"] = tool_match.group(1)
            
            args_match = re.search(r'Arguments:\s*(\{{.*?\}})', content, re.DOTALL)
            if args_match:
                try:
                    tool_info["tool_arguments"] = json.loads(args_match.group(1))
                except:
                    pass
        
        if tool_call_idx + 1 < len(contexts):
            tool_result = contexts[tool_call_idx + 1]
            if tool_result.get("role") == "tool_result":
                tool_info["tool_result"] = tool_result.get("content", "")
    
    return tool_info


def extract_experiences_with_llm(
    contexts_data: Dict[str, Any],
    model_name: str = "openai/gpt-5",
    issue_id: Optional[str] = None,
    issue_description: Optional[str] = None
) -> List[Experience]:
    """
    Extract experiences from contexts using LLM as a Judge
    
    Args:
        contexts_data: Dictionary containing issue_id and contexts
        model_name: Model name to use
        issue_id: Issue ID (if not provided, extracted from data)
        issue_description: Issue description (if not provided, extracted from messages)
        
    Returns:
        List of experience records
    """
    contexts = contexts_data.get("contexts", [])
    if not contexts:
        return []
    
    if not issue_id:
        issue_id = contexts_data.get("issue_id", "unknown")
    
    if not issue_description:
        for ctx in contexts:
            if ctx.get("role") == "user":
                content = ctx.get("content", "")
                problem_pattern = r'\[Problem statement\]:\s*(.*?)(?=\n\n|\Z)'
                match = re.search(problem_pattern, content, re.DOTALL)
                if match:
                    desc = match.group(1).strip()
                    issue_description = desc[:500] + "..." if len(desc) > 500 else desc
                    break
    
    if not issue_description:
        issue_description = "No description available"
    
    conversation_text = format_conversation_for_judge(contexts)
    
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        issue_id=issue_id,
        issue_description=issue_description,
        conversation_text=conversation_text
    )
    
    try:
        response = litellm.completion(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=1.0
        )
        llm_output = response.choices[0].message.content  # type: ignore
        
    except Exception as e:
        print(f"Error calling LLM: {e}")
        import traceback
        traceback.print_exc()
        return []
    
    experiences = []
    
    json_match = re.search(r'```json\s*(.*?)\s*```', llm_output, re.DOTALL)
    if json_match:
        json_str = json_match.group(1).strip()
    else:
        start_idx = llm_output.find('[')
        end_idx = llm_output.rfind(']')
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_str = llm_output[start_idx:end_idx+1]
        else:
            json_str = llm_output.strip()
    
    try:
        extracted_experiences = json.loads(json_str)
        
        if not isinstance(extracted_experiences, list):
            if isinstance(extracted_experiences, dict):
                extracted_experiences = [extracted_experiences]
            else:
                print(f"  Warning: LLM returned non-list, non-dict: {type(extracted_experiences)}")
                return []
        
        tool_call_indices = [i for i, ctx in enumerate(contexts) if ctx.get("role") == "tool_call"]
        
        for idx, exp_data in enumerate(extracted_experiences):
            if not isinstance(exp_data, dict):
                continue
            
            exp_idx = min(idx, len(tool_call_indices) - 1) if tool_call_indices else 0
            tool_call_idx = tool_call_indices[exp_idx] if tool_call_indices else 0
            
            tool_info = extract_tool_info_from_contexts(contexts, tool_call_idx)
            
            task_summary = exp_data.get("task_summary")
            if not task_summary:
                print(f"  Warning: Missing task_summary in experience {idx}")
                continue
            if not isinstance(task_summary, str):
                print(f"  Warning: task_summary is not a string: {type(task_summary)}")
                try:
                    task_summary = str(task_summary).strip()
                    if not task_summary:
                        continue
                except:
                    continue
            
            confidence = exp_data.get("confidence", 50)
            if not isinstance(confidence, (int, float)):
                confidence = 50
            confidence = max(1, min(100, int(confidence)))
            
            output = exp_data.get("output", "")
            if not isinstance(output, str):
                output = str(output) if output else ""
            if len(output) > 1000:
                output = output[:1000] + "..."
            
            issue_desc = issue_description[:500] + "..." if len(issue_description) > 500 else issue_description
            
            try:
                exp = Experience(
                    issue_id=issue_id,
                    issue_description=issue_desc,
                    task_summary=task_summary,
                    confidence=confidence,
                    output=output,
                    metadata={
                        "tool_name": tool_info["tool_name"],
                        "tool_arguments": tool_info["tool_arguments"],
                        "step_index": tool_info["step_index"],
                        "extraction_method": "llm_judge",
                        "reasoning": exp_data.get("reasoning", ""),
                    }
                )
                experiences.append(exp)
            except Exception as exp_error:
                print(f"  Warning: Failed to create experience for step {idx}: {exp_error}")
                continue
            
    except json.JSONDecodeError as e:
        print(f"  Error parsing LLM output as JSON: {e}")
        print(f"  JSON string (first 500 chars):\n{json_str[:500]}")
        print(f"  Full LLM output (first 1000 chars):\n{llm_output[:1000]}")
        return []
    except Exception as e:
        print(f"  Error processing experiences: {e}")
        import traceback
        traceback.print_exc()
        return []
    
    return experiences


def load_filter_ids(filter_file: str | None) -> set[str]:
    """Load issue IDs to filter from file"""
    if not filter_file or not Path(filter_file).exists():
        return set()
    
    filter_ids = set()
    with Path(filter_file).open(encoding='utf-8') as f:
        for line in f:
            issue_id = line.strip()
            if issue_id:
                filter_ids.add(issue_id)
    
    print(f"Loaded {len(filter_ids)} filter IDs from {filter_file}")
    return filter_ids


def process_single_issue(
    line_data: Tuple[int, str, Dict[str, Any], str, set[str], Optional[Path], Lock]
) -> Tuple[int, str, List[Experience], Optional[str]]:
    """
    Helper function to process a single issue (for parallel processing)
    
    Returns:
        (line_num, issue_id, experiences, error_message)
    """
    line_num, line, contexts_data, model_name, filter_ids, output_path, write_lock = line_data
    
    try:
        issue_id = contexts_data.get("issue_id", "unknown")
        
        if filter_ids and issue_id not in filter_ids:
            return (line_num, issue_id, [], None)
        
        experiences = extract_experiences_with_llm(
            contexts_data,
            model_name=model_name
        )
        
        if output_path and experiences:
            with write_lock:
                with output_path.open('a', encoding='utf-8') as out_f:
                    for exp in experiences:
                        out_f.write(exp.to_json() + '\n')
        
        return (line_num, issue_id, experiences, None)
        
    except Exception as e:
        error_msg = str(e)
        return (line_num, contexts_data.get("issue_id", "unknown"), [], error_msg)


def extract_from_jsonl(
    jsonl_path: str,
    model_name: str = "openai/gpt-4o",
    output_file: Optional[str] = None,
    filter_ids_file: Optional[str] = None,
    num_workers: int = 1
) -> List[Experience]:
    """
    Extract experiences from JSONL file
    
    Args:
        jsonl_path: JSONL file path
        model_name: Model name to use
        output_file: Output file path (JSONL format)
        filter_ids_file: Filter file path, only extract issue IDs listed in the file
        num_workers: Number of parallel workers (default 1, single-threaded)
        
    Returns:
        List of experience records
    """
    all_experiences = []
    
    filter_ids = load_filter_ids(filter_ids_file) if filter_ids_file else set()
    
    print(f"Reading contexts from: {jsonl_path}")
    if filter_ids_file:
        print(f"Using filter file: {filter_ids_file}")
    if filter_ids:
        print(f"Will only process {len(filter_ids)} specified issue IDs")
    print(f"Using model: {model_name}")
    print(f"Workers: {num_workers}")
    print("=" * 60)
    
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    output_path = None
    write_lock = Lock()
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text('', encoding='utf-8')
        print(f"Output file initialized: {output_path}")
    
    tasks = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                contexts_data = json.loads(line)
                issue_id = contexts_data.get("issue_id", "unknown")
                
                if filter_ids and issue_id not in filter_ids:
                    skipped_count += 1
                    continue
                
                tasks.append((line_num, line, contexts_data, model_name, filter_ids, output_path, write_lock))
                
            except json.JSONDecodeError as e:
                print(f"  ✗ Error parsing JSON on line {line_num}: {e}")
                error_count += 1
    
    if not tasks:
        print("No tasks to process.")
        return []
    
    print(f"Total tasks to process: {len(tasks)}")
    print("=" * 60)
    
    start_time = time.time()
    
    if num_workers == 1:
        for line_num, line, contexts_data, model_name, filter_ids, output_path, write_lock in tasks:
            issue_id = contexts_data.get("issue_id", "unknown")
            print(f"\n[{line_num}] Processing: {issue_id}...")
            
            result = process_single_issue((line_num, line, contexts_data, model_name, filter_ids, output_path, write_lock))
            line_num, issue_id, experiences, error_msg = result
            
            if error_msg:
                print(f"  ✗ Error: {error_msg}")
                error_count += 1
            elif experiences:
                all_experiences.extend(experiences)
                print(f"  ✓ Extracted {len(experiences)} experiences")
                if output_path:
                    print(f"  💾 Saved {len(experiences)} experiences to file")
                processed_count += 1
            else:
                print(f"  ⚠ No experiences extracted")
                processed_count += 1
    else:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_task = {
                executor.submit(process_single_issue, task): task 
                for task in tasks
            }
            
            completed = 0
            for future in as_completed(future_to_task):
                completed += 1
                line_num, issue_id, experiences, error_msg = future.result()
                
                if error_msg:
                    print(f"[{line_num}] {issue_id}: ✗ Error - {error_msg}")
                    error_count += 1
                elif experiences:
                    all_experiences.extend(experiences)
                    print(f"[{line_num}] {issue_id}: ✓ Extracted {len(experiences)} experiences [{completed}/{len(tasks)}]")
                    processed_count += 1
                else:
                    print(f"[{line_num}] {issue_id}: ⚠ No experiences extracted [{completed}/{len(tasks)}]")
                    processed_count += 1
    
    elapsed_time = time.time() - start_time
    
    print("=" * 60)
    print(f"Processing complete!")
    print(f"  Success: {processed_count} issues")
    if filter_ids:
        print(f"  Skipped: {skipped_count} issues")
    print(f"  Errors: {error_count} issues")
    print(f"  Total experiences: {len(all_experiences)}")
    print(f"  Time elapsed: {elapsed_time:.2f} seconds")
    if processed_count > 0:
        print(f"  Average time per issue: {elapsed_time/processed_count:.2f} seconds")
    
    if output_path:
        print(f"\nAll experiences saved to: {output_path}")
    
    return all_experiences


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract experiences from merged_contexts.jsonl using LLM as a Judge"
    )
    parser.add_argument(
        "jsonl_path",
        type=str,
        help="Path to merged_contexts.jsonl file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-4o",
        help="Model name to use (default: openai/gpt-4o)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (JSONL format). Default: extracted_experiences_llm.jsonl in same directory"
    )
    parser.add_argument(
        "--filter-ids",
        type=str,
        default=None,
        help="File containing issue IDs to filter (one per line)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1, single-threaded)"
    )
    
    args = parser.parse_args()
    
    if not args.output:
        jsonl_path = Path(args.jsonl_path)
        args.output = str(jsonl_path.parent / "extracted_experiences_llm.jsonl")
    
    experiences = extract_from_jsonl(
        args.jsonl_path,
        model_name=args.model,
        output_file=args.output,
        filter_ids_file=args.filter_ids,
        num_workers=args.workers
    )
    
    if experiences:
        issue_ids = set(exp.issue_id for exp in experiences)
        avg_confidence = sum(exp.confidence for exp in experiences) / len(experiences)
        
        print("\nStatistics:")
        print(f"  Total experiences: {len(experiences)}")
        print(f"  Unique issues: {len(issue_ids)}")
        print(f"  Average confidence: {avg_confidence:.2f}")


if __name__ == "__main__":
    main()

