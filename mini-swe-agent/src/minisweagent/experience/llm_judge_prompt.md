# LLM as a Judge Prompt for Experience Extraction

## Overview
This prompt is designed to guide Claude (or other LLMs) to extract valuable experiences from agent interaction dialogues. The LLM acts as a judge to identify meaningful steps and extract structured experience records.

## Full Prompt Template

```
You are an expert judge tasked with extracting valuable experiences from agent interaction dialogues. Your goal is to identify meaningful steps that contribute to solving programming tasks and extract structured experience records.

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
  {
    "task_summary": "Brief summary of what this step accomplishes (max 200 chars)",
    "confidence": 85,
    "output": "Relevant output from the command execution (max 1000 chars, truncate if needed)",
    "reasoning": "Why this step is valuable and what makes it significant"
  }
]
```

## Evaluation Criteria

### 1. Task Summary
Create a concise, descriptive summary (format: "Task Type: Brief description")
- **Task Types**: 
  - "Find Files" - Searching for files or directories
  - "Read Code" - Reading source code files
  - "Analyze Logic" - Understanding code logic and flow
  - "Modify Code" - Editing or creating code files
  - "Run Tests" - Executing test suites
  - "Debug Issue" - Investigating errors or bugs
  - "Reproduce Issue" - Creating scripts to reproduce problems
  - "Verify Fix" - Testing solutions
  - "Explore Codebase" - Understanding project structure
- **Description**: Should capture the key intent and action of the step

### 2. Confidence Score (1-100)
Assess based on multiple factors:
- **Exit Status**: If issue was "Submitted" (successful), increase confidence by 20-30 points
- **Command Success**: If returncode == 0, increase confidence by 10-15 points
- **Output Quality**: If output provides useful information, increase confidence by 5-10 points
- **Step Significance**: If step directly contributes to solution, increase confidence by 10-20 points
- **Base Confidence Guidelines**:
  - 50-60: Exploratory steps (e.g., initial exploration, trying different approaches)
  - 70-85: Successful actions (e.g., reading relevant code, understanding the issue)
  - 85-95: Final solution steps (e.g., implementing fix, verifying solution)
  - 95-100: Critical steps that directly solve the problem

### 3. Output
Extract relevant output from the next user message (command execution result)
- Focus on useful information, not error messages or empty outputs
- Include key findings, code snippets, or results
- Truncate to 1000 characters if too long
- If output is empty or unhelpful, use empty string

### 4. Reasoning (Optional)
Explain why this step is valuable (helps with quality assessment)
- What makes this step significant?
- How does it contribute to solving the issue?
- What insights or information does it provide?

## Filtering Guidelines
**Include** steps that:
- Show clear problem-solving progression
- Execute meaningful commands with useful outputs
- Contribute to understanding or fixing the issue
- Demonstrate effective reasoning and planning

**Skip** steps that:
- Are trivial (e.g., simple `ls` without clear purpose)
- Have empty or unhelpful outputs
- Are redundant or repetitive
- Don't contribute to problem-solving

## Example

### Input:
```
Issue ID: psf__requests-1963
Issue Description: Session.resolve_redirects copies original request for all subsequent requests, can cause incorrect method selection...

Messages:
[SYSTEM 1]
You are a helpful assistant...

[USER 1]
<pr_description>
Session.resolve_redirects copies the original request...
</pr_description>

[ASSISTANT 1]
THOUGHT: I need to understand the codebase structure first to locate the relevant files...
```bash
ls -la
```

[USER 2]
<returncode>0</returncode>
<output>
total 140
drwxrwxrwx 1 root root  4096 Nov 22  2024 .
...
-rw-r--r-- 1 root root  20002 Nov 22  2024 sessions.py
...
</output>

[ASSISTANT 2]
THOUGHT: Perfect! I can see there's a sessions.py file which likely contains the Session class...
```bash
cat requests/sessions.py
```

[USER 3]
<returncode>0</returncode>
<output>
class SessionRedirectMixin(object):
    def resolve_redirects(self, resp, req, ...):
        i = 0
        while resp.is_redirect:
            prepared_request = req.copy()  # <-- This is the bug!
            ...
```

### Output:
```json
[
  {
    "task_summary": "Explore Codebase: List directory structure to locate relevant files",
    "confidence": 70,
    "output": "total 140\ndrwxrwxrwx 1 root root  4096 Nov 22  2024 .\n...\n-rw-r--r-- 1 root root  20002 Nov 22  2024 sessions.py",
    "reasoning": "This step identifies the codebase structure and locates the key file containing the bug"
  },
  {
    "task_summary": "Read Code: Examine sessions.py to locate resolve_redirects method",
    "confidence": 85,
    "output": "class SessionRedirectMixin(object):\n    def resolve_redirects(self, resp, req, ...):\n        i = 0\n        while resp.is_redirect:\n            prepared_request = req.copy()  # <-- This is the bug!",
    "reasoning": "This step identifies the exact location of the bug, which is crucial for fixing it"
  }
]
```

## Notes
- Be selective: Quality over quantity. Not every step needs to be extracted.
- Focus on learning value: Extract steps that would be useful for future similar tasks.
- Maintain context: Ensure extracted experiences capture the problem-solving flow.
- Be consistent: Use similar task types and confidence scoring across similar steps.

