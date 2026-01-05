#!/usr/bin/env python3
"""
Extract experience records from cleaned conversation files using LLM as a Judge
"""
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
import re
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

try:
    from minisweagent.experience.models import Experience
    from minisweagent.models.litellm_model import LitellmModel, LitellmModelConfig
    USE_LITELLM_MODEL = True
except ImportError:
    try:
        from models import Experience
    except ImportError:
        print("Error: Cannot import Experience model")
        sys.exit(1)
    
    try:
        import litellm  # type: ignore
        USE_LITELLM_MODEL = False
    except ImportError:
        print("Error: Please install litellm: pip install litellm")
        sys.exit(1)


JUDGE_PROMPT_TEMPLATE = """You are an expert judge tasked with extracting valuable experiences from agent interaction dialogues. Your goal is to identify meaningful steps that contribute to solving programming tasks and extract structured experience records.

## Task
Analyze the conversation history and extract experience records for each significant step where the assistant interacts with the system. Focus on steps that:
1. Show clear reasoning and planning (THOUGHT section)
2. Execute meaningful commands
3. Produce useful outputs or insights
4. Contribute to solving the issue

## Input Format
You will receive:
- issue_id: The identifier of the issue being solved
- issue_description: Brief description of the problem
- messages: A list of conversation messages (system, user, assistant)

## Output Format
For each significant assistant step, extract an experience record in JSON format. Return a JSON array of experience objects:

```json
[
  {{
    "task_summary": "Brief summary of what this step accomplishes (max 200 chars)",
    "confidence": 85,
    "output": "Relevant output from the command execution (max 1000 chars, truncate if needed)",
    "reasoning": "Why this step is valuable and what makes it significant"
  }}
]
```

## Evaluation Criteria
1. **Task Summary**: Create a concise, descriptive summary (format: "Task Type: Brief description")
   - Task types: "Find Files", "Read Code", "Analyze Logic", "Modify Code", "Run Tests", "Debug Issue", etc.
   - Description should capture the key intent and action

2. **Confidence Score** (1-100): Assess based on:
   - Exit status: If issue was "Submitted" (successful), increase confidence
   - Command success: If returncode == 0, increase confidence
   - Output quality: If output provides useful information, increase confidence
   - Step significance: If step directly contributes to solution, increase confidence
   - Base confidence: 50-60 for exploratory steps, 70-85 for successful actions, 85-95 for final solution steps

3. **Output**: Extract relevant output from the next user message (command execution result)
   - Focus on useful information, not error messages or empty outputs
   - Truncate to 1000 characters if too long

4. **Reasoning**: Explain why this step is valuable.

## Filtering Guidelines
- Skip trivial steps (e.g., simple ls without clear purpose)
- Skip steps with empty or unhelpful outputs
- Focus on steps that show problem-solving progression
- Prioritize steps that lead to understanding or fixing the issue

## Example
Input:
- issue_id: "psf__requests-1963"
- issue_description: "Session.resolve_redirects copies original request..."
- messages: [system message, user prompt, assistant response with THOUGHT and command, user output...]

Output:
```json
[
  {{
    "task_summary": "Read Code: Examine sessions.py to locate resolve_redirects method",
    "confidence": 85,
    "output": "[file content showing the method implementation]",
    "reasoning": "This step identifies the exact location of the bug, which is crucial for fixing it"
  }}
]
```

Now analyze the following conversation and extract experience records:

Issue ID: {issue_id}
Issue Description: {issue_description}

Messages:
{conversation_text}

Extract all valuable experience records as a JSON array:"""


def format_conversation_for_judge(messages: List[Dict]) -> str:
    """Format message list into text suitable for LLM reading"""
    formatted_lines = []
    
    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        
        if len(content) > 5000:
            content = content[:5000] + "\n... [truncated]"
        
        formatted_lines.append(f"[{role.upper()} {i+1}]")
        formatted_lines.append(content)
        formatted_lines.append("")
    
    return "\n".join(formatted_lines)


def extract_experiences_with_llm(
    cleaned_traj_path: str,
    model_name: str = "openai/gpt-5",
    issue_id: Optional[str] = None,
    issue_description: Optional[str] = None
) -> List[Experience]:
    """
    Extract experiences from cleaned trajectory files using LLM as a Judge
    
    Args:
        cleaned_traj_path: Path to cleaned JSON file (*_cleaned.json)
        model_name: Model name to use
        issue_id: Issue ID (if not provided, extracted from filename)
        issue_description: Issue description (if not provided, extracted from messages)
        
    Returns:
        List of experience records
    """
    with open(cleaned_traj_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    messages = data.get("messages", [])
    if not messages:
        return []
    
    if not issue_id:
        filename = os.path.basename(cleaned_traj_path)
        issue_id = filename.replace("_cleaned.json", "")
    
    if not issue_description:
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                pr_pattern = r'<pr_description>\s*(.*?)\s*</pr_description>'
                match = re.search(pr_pattern, content, re.DOTALL)
                if match:
                    desc = match.group(1).strip()
                    desc = re.sub(r'^Consider the following PR description:\s*', '', desc, flags=re.IGNORECASE)
                    issue_description = desc[:500] + "..." if len(desc) > 500 else desc
                    break
    
    if not issue_description:
        issue_description = "No description available"
    
    conversation_text = format_conversation_for_judge(messages)
    
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        issue_id=issue_id,
        issue_description=issue_description,
        conversation_text=conversation_text
    )
    
    try:
        if USE_LITELLM_MODEL:
            model = LitellmModel(
                model_name=model_name,
                model_kwargs={"temperature": 0.0}
            )
            
            response = model.query([{"role": "user", "content": prompt}])
            llm_output = response["content"]
        else:
            import litellm  # type: ignore
            response = litellm.completion(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
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
        
        assistant_indices = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
        
        for idx, exp_data in enumerate(extracted_experiences):
            if not isinstance(exp_data, dict):
                continue
            
            exp_idx = min(idx, len(assistant_indices) - 1) if assistant_indices else 0
            assistant_idx = assistant_indices[exp_idx] if assistant_indices else 0
            
            assistant_msg = messages[assistant_idx] if assistant_idx < len(messages) else None
            command = None
            
            if assistant_msg:
                content = assistant_msg.get("content", "")
                bash_pattern = r'```bash\s*\n(.*?)\n```'
                cmd_match = re.search(bash_pattern, content, re.DOTALL)
                if cmd_match:
                    command = cmd_match.group(1).strip()
            
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
                        "command": command,
                        "step_index": assistant_idx,
                        "extraction_method": "llm_judge",
                        "reasoning": exp_data.get("reasoning", ""),
                        "traj_file": os.path.basename(cleaned_traj_path)
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


def extract_from_directory(
    directory: str,
    pattern: str = "*_cleaned.json",
    model_name: str = "openai/gpt-5",
    output_file: Optional[str] = None,
    filter_ids_file: Optional[str] = None
) -> List[Experience]:
    """
    Extract experiences from all cleaned files in directory
    
    Args:
        directory: Directory path
        pattern: File matching pattern
        model_name: Model name to use
        output_file: Output file path (JSONL format)
        filter_ids_file: Filter file path, only extract issue IDs listed in the file
        
    Returns:
        List of experience records
    """
    directory_path = Path(directory)
    all_experiences = []
    
    filter_ids = load_filter_ids(filter_ids_file) if filter_ids_file else set()
    
    cleaned_files = list(directory_path.rglob(pattern))
    
    print(f"Found {len(cleaned_files)} cleaned trajectory files")
    if filter_ids_file:
        print(f"Using filter file: {filter_ids_file}")
    if filter_ids:
        print(f"Will only process {len(filter_ids)} specified issue IDs")
    print(f"Using model: {model_name}")
    print("=" * 60)
    
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    output_path = None
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text('', encoding='utf-8')
        print(f"Output file initialized: {output_path}")
    
    for cleaned_file in cleaned_files:
        instance_id = cleaned_file.parent.name
        
        if filter_ids and instance_id not in filter_ids:
            skipped_count += 1
            continue
        
        try:
            print(f"Processing: {cleaned_file.name}...")
            experiences = extract_experiences_with_llm(
                str(cleaned_file),
                model_name=model_name
            )
            
            if experiences:
                all_experiences.extend(experiences)
                print(f"  ✓ Extracted {len(experiences)} experiences")
                
                if output_path:
                    with output_path.open('a', encoding='utf-8') as f:
                        for exp in experiences:
                            f.write(exp.to_json() + '\n')
                    print(f"  💾 Saved {len(experiences)} experiences to file")
            else:
                print(f"  ⚠ No experiences extracted")
            
            processed_count += 1
            
        except Exception as e:
            print(f"  ✗ Error processing {cleaned_file.name}: {e}")
            import traceback
            if "task_summary" in str(e) or isinstance(e, (KeyError, TypeError, ValueError)):
                print(f"    Traceback:")
                traceback.print_exc()
            error_count += 1
    
    print("=" * 60)
    print(f"Processing complete!")
    print(f"  Success: {processed_count} files")
    if filter_ids:
        print(f"  Skipped: {skipped_count} files")
    print(f"  Errors: {error_count} files")
    print(f"  Total experiences: {len(all_experiences)}")
    
    if output_path:
        print(f"\nAll experiences saved to: {output_path}")
    
    return all_experiences


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract experiences from cleaned trajectory files using LLM as a Judge"
    )
    parser.add_argument(
        "directory",
        type=str,
        help="Directory containing cleaned trajectory files (*_cleaned.json)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-5",
        help="Model name to use (default: openai/gpt-5)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (JSONL format). Default: extracted_experiences_llm.jsonl in directory"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*_cleaned.json",
        help="File pattern to match (default: *_cleaned.json)"
    )
    parser.add_argument(
        "--filter-ids",
        type=str,
        default=None,
        help="File containing issue IDs to filter (one per line). Default: exp_ids.txt in script directory"
    )
    
    args = parser.parse_args()
    
    if not args.filter_ids:
        default_filter_file = Path(__file__).parent / "exp_ids.txt"
        if default_filter_file.exists():
            args.filter_ids = str(default_filter_file)
    
    if not args.output:
        args.output = str(Path(args.directory) / "extracted_experiences_llm.jsonl")
    
    experiences = extract_from_directory(
        args.directory,
        pattern=args.pattern,
        model_name=args.model,
        output_file=args.output,
        filter_ids_file=args.filter_ids
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

