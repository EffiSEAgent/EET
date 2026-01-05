"""Agent with experience retrieval integrated into prompt generation."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from jinja2 import StrictUndefined, Template

from minisweagent.agents.default import AgentConfig, DefaultAgent
from minisweagent.experience.store import ExperienceStore
from minisweagent.experience.models import Experience
from minisweagent.utils.log import logger


@dataclass
class ExperienceRetrievalAgentConfig(AgentConfig):
    """Configuration for ExperienceRetrievalAgent."""
    experience_store_path: Optional[str] = None
    """Path to the experience store JSONL file."""
    experience_top_k: int = 1
    """Number of top experiences to retrieve."""
    experience_min_similarity: float = 0.15
    """Minimum similarity threshold for experience retrieval."""
    experience_use_tfidf: bool = True
    """Whether to use TF-IDF for similarity calculation."""
    experience_search_by_issue: bool = True
    """Whether to search by issue similarity (True) or task similarity (False)."""
    experience_template: str = """
<retrieved_experiences>
{% if experiences %}
Here are some relevant experiences from similar tasks that might help you:

{% for exp, similarity in experiences %}
## Experience {{ loop.index }}
**Issue Description:** {{ exp.issue_description }}
**Task Summary:** {{ exp.task_summary }}
---
{% endfor %}
</retrieved_experiences>
{% endif %}
"""
    """Template for formatting retrieved experiences in the prompt."""
    confidence_check_enabled: bool = True
    """Whether to enable confidence check for early submission."""


class ExperienceRetrievalAgent(DefaultAgent):
    """Agent that retrieves and integrates relevant experiences into prompts."""
    
    def __init__(
        self,
        model,
        env,
        *,
        config_class=ExperienceRetrievalAgentConfig,
        **kwargs
    ):
        super().__init__(model, env, config_class=config_class, **kwargs)
        self.config: ExperienceRetrievalAgentConfig = self.config
        
        # Initialize experience store if path is provided
        self.experience_store: Optional[ExperienceStore] = None
        if self.config.experience_store_path:
            store_path = Path(self.config.experience_store_path)
            # Try resolving relative to current working directory first
            if not store_path.is_absolute():
                # If relative path doesn't exist, try relative to the experience module
                if not store_path.exists():
                    experience_dir = Path(__file__).parent.parent / "experience"
                    alt_path = experience_dir / store_path
                    if alt_path.exists():
                        store_path = alt_path
                    else:
                        # Try just the filename in experience directory
                        alt_path = experience_dir / store_path.name
                        if alt_path.exists():
                            store_path = alt_path
            
            if store_path.exists():
                self.experience_store = ExperienceStore(storage_path=str(store_path))
            else:
                import warnings
                warnings.warn(f"Experience store path does not exist: {store_path}")
        
        # Store original template to restore if needed
        self._original_instance_template = self.config.instance_template
        
        # Track whether experiences were retrieved for this run
        self._has_retrieved_experiences = False
        
        # Initialize workflow progress tracking for confidence check
        self.workflow_progress = {
            'steps': 0,
            'has_code_changes': False,
            'test_passed': False,
            'last_confidence_check': None,
            'confidence_score': None
        }
    
    def _extract_project_name(self, issue_id: str) -> Optional[str]:
        """
        Extract project name from issue_id.
        
        Args:
            issue_id: Issue ID in format like "pylint-dev__pylint-5859" or "django__django-11119.traj"
            
        Returns:
            Project name (e.g., "pylint-dev__pylint" or "django__django") or None if format is invalid
        """
        if not issue_id:
            return None
        
        # Remove .traj suffix if present
        issue_id_clean = issue_id.replace('.traj', '')
        
        # Issue ID format: "project-name-issue-number" or "project__project-issue-number"
        # Extract project name by removing the last dash and number part
        parts = issue_id_clean.rsplit('-', 1)
        if len(parts) == 2 and parts[1].isdigit():
            return parts[0]
        return None
    
    def retrieve_experiences(self, task: str, issue_id: Optional[str] = None) -> list[tuple[Experience, float]]:
        """
        Retrieve relevant experiences based on the task.
        
        Args:
            task: The task description/problem statement
            issue_id: Optional issue ID to also search by issue_id
            
        Returns:
            List of (Experience, similarity_score) tuples
        """
        if not self.experience_store:
            return []
        
        # Extract project name from current issue_id to filter cross-project matches
        current_project_name = None
        if issue_id:
            current_project_name = self._extract_project_name(issue_id)
        
        experiences = []
        
        # Then, search by similarity
        if self.config.experience_search_by_issue:
            # Search by issue description similarity
            similarity_results = self.experience_store.search_by_issue_similarity(
                query=task,
                top_k=self.config.experience_top_k * 2,  # Retrieve more initially to account for filtering
                min_similarity=self.config.experience_min_similarity,
                use_tfidf=self.config.experience_use_tfidf
            )
        else:
            # Search by task summary similarity
            similarity_results = self.experience_store.search_by_task_similarity(
                query=task,
                top_k=self.config.experience_top_k * 2,  # Retrieve more initially to account for filtering
                min_similarity=self.config.experience_min_similarity,
                use_tfidf=self.config.experience_use_tfidf
            )
        
        # Add similarity-based results, avoiding duplicates and filtering by project name
        existing_issue_ids = {exp.issue_id for exp, _ in experiences}
        for exp, similarity in similarity_results:
            # Skip if already added
            if exp.issue_id in existing_issue_ids:
                continue
            
            # Filter by project name if current project name is available
            if current_project_name:
                exp_project_name = self._extract_project_name(exp.issue_id)
                if exp_project_name != current_project_name:
                    continue  # Skip experiences from different projects
            
            experiences.append((exp, similarity))
            existing_issue_ids.add(exp.issue_id)
        
        # Sort by similarity (descending) and limit to top_k
        experiences.sort(key=lambda x: x[1], reverse=True)
        return experiences[:self.config.experience_top_k]
    
    def format_experiences(self, experiences: list[tuple[Experience, float]]) -> str:
        """
        Format retrieved experiences using the template.
        
        Args:
            experiences: List of (Experience, similarity_score) tuples
            
        Returns:
            Formatted string to include in prompt
        """
        if not experiences:
            return ""
        
        # Process experiences to handle task_summary as list
        processed_experiences = []
        for exp, similarity in experiences:
            # Convert task_summary list to string if needed
            task_summary = exp.task_summary
            if isinstance(task_summary, list):
                task_summary = "\n".join(f"- {task}" for task in task_summary)
            
            # Create a simple object with only issue_description and task_summary
            processed_exp = type('Experience', (), {
                'issue_description': exp.issue_description,
                'task_summary': task_summary
            })()
            processed_experiences.append((processed_exp, similarity))
        
        template = Template(
            self.config.experience_template,
            undefined=StrictUndefined
        )
        return template.render(experiences=processed_experiences)
    
    def render_template(self, template: str, **kwargs) -> str:
        """Override to include experiences in template variables."""
        template_vars = self.env.get_template_vars() | self.model.get_template_vars()
        template_vars.update(self.extra_template_vars)
        template_vars.update(kwargs)
        
        # Include experiences if available
        if "experiences" not in template_vars and hasattr(self, "_current_experiences"):
            template_vars["experiences"] = self._current_experiences
        
        return Template(template, undefined=StrictUndefined).render(**template_vars)
    
    def _is_key_step(self, action: str, output: dict) -> bool:
        """
        Determine if this is a key step that warrants confidence checking.
        
        Key steps include:
        - Actual code modifications (git add, sed edits)
        - Test execution (regardless of pass/fail)
        - Verification commands (with precise matching)
        - Test passed (based on workflow progress)
        """
        action_lower = action.lower()
        
        # 1. Actual code modifications (high confidence)
        if 'git add' in action_lower:
            return True
        elif 'sed -i' in action_lower:
            return True
        
        # 2. Test execution (regardless of pass/fail)
        if any(cmd in action_lower for cmd in ['pytest', 'python -m pytest']):
            return True
        
        # 3. Test scripts (more precise matching)
        if 'python' in action_lower:
            # Check if it's running a test/verify script
            if any(pattern in action_lower for pattern in ['test_', '_test.py', '/test', 'verify', 'check']):
                # Make sure it's executing, not just viewing
                if not any(view_cmd in action_lower for view_cmd in ['cat', 'grep', 'less', 'more', 'head', 'tail']):
                    return True
        
        # 4. Check workflow progress for additional triggers
        # Note: _assess_workflow_progress is already called in get_observation,
        # but we call it here to ensure state is up-to-date
        self._assess_workflow_progress(action, output)
        if self.workflow_progress.get('test_passed', False):
            return True
        
        return False
    
    def _assess_workflow_progress(self, action: str, output: dict) -> None:
        """Update workflow progress based on action and output."""
        action_lower = action.lower()
        returncode = output.get('returncode', 1)
        output_text = output.get('output', '')
        output_text_lower = output_text.lower()
        
        # Track code changes - check for actual modifications
        if 'git diff' in action_lower or 'git status' in action_lower:
            # Check if there are actual changes in the output
            if 'diff --git' in output_text or 'modified:' in output_text or '+++' in output_text:
                self.workflow_progress['has_code_changes'] = True
        elif 'sed -i' in action_lower:
            # sed -i modifies files directly
            self.workflow_progress['has_code_changes'] = True
        elif 'git add' in action_lower:
            # git add indicates files were modified
            self.workflow_progress['has_code_changes'] = True
        
        # Track test results - only mark as passed if test actually succeeded
        if any(cmd in action_lower for cmd in ['pytest', 'python -m pytest']):
            if returncode == 0:
                # Check for pytest success indicators
                if any(indicator in output_text_lower for indicator in ['passed', 'test passed', ' passed ']):
                    # Make sure it's not showing failures
                    if 'failed' not in output_text_lower and 'error' not in output_text_lower:
                        self.workflow_progress['test_passed'] = True
        elif 'python' in action_lower and ('test' in action_lower or 'verify' in action_lower):
            if returncode == 0:
                # Check for general success indicators in script output
                if any(indicator in output_text_lower for indicator in ['passed', 'success', 'ok', 'test passed']):
                    if 'failed' not in output_text_lower and 'error' not in output_text_lower:
                        self.workflow_progress['test_passed'] = True
    
    def _should_exclude_file(self, file_path: str) -> bool:
        """
        Check if a file should be excluded from submission.
        
        Excludes:
        - Test files (test_*.py, *_test.py, reproduce_issue*.py)
        - Debug files (debug_*.py)
        - Temporary scripts
        """
        file_path_lower = file_path.lower()
        # Exclude test files
        if any(pattern in file_path_lower for pattern in [
            'test_', '_test.py', 'reproduce_issue', 'reproduce.py'
        ]):
            return True
        # Exclude debug files
        if 'debug_' in file_path_lower and file_path_lower.endswith('.py'):
            return True
        return False
    
    def _get_source_files_to_add(self) -> list[str]:
        """
        Get list of source files that should be added (excluding test/debug files).
        
        Returns:
            List of file paths that are safe to add to git
        """
        try:
            # Execute git status to see what files have changed
            status_output = self.execute_action({'action': 'git status --porcelain'})
            if status_output.get('returncode', 1) != 0:
                return []
            
            output_text = status_output.get('output', '')
            source_files = []
            
            for line in output_text.split('\n'):
                line = line.strip()
                if not line:
                    continue
                # git status --porcelain format: XY filename (or ?? for untracked)
                # X = staged status, Y = unstaged status
                if len(line) < 3:
                    continue
                
                # Handle untracked files (??)
                if line.startswith('??'):
                    file_path = line[3:].strip()
                else:
                    # Extract filename (skip status codes)
                    file_path = line[3:].strip()
                
                # Handle filenames with spaces (rare but possible)
                # For simplicity, assume filename starts after position 3
                if not file_path:
                    continue
                
                # Only include Python source files
                if not file_path.endswith('.py'):
                    continue
                
                # Exclude test/debug files
                if self._should_exclude_file(file_path):
                    continue
                
                # Include modified/new files (not deleted, which start with 'D')
                status_code = line[:2]
                if status_code[0] == 'D' or status_code[1] == 'D':
                    continue  # Skip deleted files
                
                source_files.append(file_path)
            
            return source_files
        except Exception:
            # If anything goes wrong, return empty list (fail gracefully)
            return []
    
    def _validate_fix_logic(self, diff_output: str) -> tuple[bool, str]:
        """
        Validate that the fix logic looks reasonable.
        
        Checks for common issues:
        - Missing parameters in function calls
        - Incorrect return types
        - Test files being included
        
        Returns:
            Tuple of (is_valid, reason)
        """
        diff_lower = diff_output.lower()
        
        # Check if test files are in the diff
        if any(pattern in diff_lower for pattern in [
            'test_', '_test.py', 'reproduce_issue', 'debug_'
        ]):
            return False, "Submission contains test files or debug files, which should not be submitted"
        
        # Check for suspicious patterns (can be expanded)
        # Missing obj parameter in function calls (as seen in django-11149)
        if 'has_add_permission(request)' in diff_output and 'has_view_permission' in diff_output:
            if 'obj' not in diff_output.split('has_add_permission(request)')[0][-100:]:
                return False, "Fix may be missing required parameters (e.g., obj parameter)"
        
        return True, ""
    
    def _extract_confidence_score(self, text: str) -> Optional[int]:
        """
        Extract confidence score from LLM response.
        
        Only matches the strict format: CONFIDENCE_SCORE: <number>
        
        Returns:
            Confidence score (0-100) or None if not found
        """
        # Only match the strict format: CONFIDENCE_SCORE: <number>
        match = re.search(r'CONFIDENCE_SCORE:\s*(\d{1,3})', text, re.IGNORECASE)
        if match:
            score = int(match.group(1))
            if 0 <= score <= 100:
                return score
        
        return None
    
    def _get_confidence_check_prompt(self) -> str:
        """
        Generate a prompt asking LLM to assess its own confidence score (0-100).
        This is separate from submission prompting.
        """
        if not self.config.confidence_check_enabled:
            return ""
        
        prompt = """
<confidence_check>
## Confidence Self-Assessment Request

You've reached a key milestone in your task. Please take a moment to assess your progress based on the **complete context** of your work so far, including:

- The issue description and requirements
- All the commands you've executed and their results
- Code changes you've made (if any)
- Tests or verification you've run (if any)
- Your understanding of the problem and solution

**Please provide your confidence assessment:**

**Confidence Score (0-100):** 
Based on your understanding of the entire task context, how confident are you that the task is complete?

Consider:
- Have you addressed the core issue described in the PR?
- Are your changes appropriate and complete?
- Have you verified that your solution works correctly?
- Is the solution robust and handles edge cases?
- Have you checked that your fix logic is correct? (e.g., function parameters, return types, logic flow)

**Please respond with your confidence score using this exact format:**
```
CONFIDENCE_SCORE: <number>
```

Where `<number>` is an integer between 0 and 100.

Example: `CONFIDENCE_SCORE: 85`

After providing your score, continue with your next command as usual. The system will evaluate your score and provide guidance accordingly.
</confidence_check>
"""
        return prompt
    
    def _get_submission_prompt(self, confidence_score: int) -> str:
        """
        Generate submission prompt based on confidence score.
        
        Args:
            confidence_score: The confidence score (0-100) from LLM
            
        Returns:
            Submission prompt if score is high enough, empty string otherwise
        """
        # Threshold for prompting submission
        submission_threshold = 81
        
        if confidence_score < submission_threshold:
            return ""
        
        # Check what files would be submitted
        source_files = self._get_source_files_to_add()
        files_warning = ""
        if not source_files:
            files_warning = "\n **Warning**: No source code files detected for submission. Please ensure you have modified source code files."
        elif len(source_files) > 10:
            files_warning = f"\n **Warning**: {len(source_files)} files will be submitted. Please confirm these are all necessary changes."
        
        prompt = f"""
<submission_prompt>
## Ready to Submit?

Based on your confidence score of **{confidence_score}/100**, you appear to be ready for submission.

**Before submitting, please verify:**

1. **Only source code files are being submitted** (No test files, debug scripts, or temporary files)
   - Test files: `test_*.py`, `*_test.py`, `reproduce_issue*.py`
   - Debug files: `debug_*.py`
   - These files should be created in temporary locations or excluded from git

2. **Your fix logic is correct**
   - Function parameters are correct
   - Return types match expectations
   - Logic flow handles all cases

**To submit, execute:**
```bash
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git status --porcelain | grep -E "^\?\?|^ M|^A " | grep -vE "(test_|_test\.py|reproduce_issue|debug_)" | awk '{{print $2}}' | xargs -r git add && git diff --cached
```

Or manually add only source files:
```bash
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git add <source_file1.py> <source_file2.py> && git diff --cached
```

If you need to make more changes, continue working as usual.
{files_warning}
</submission_prompt>
"""
        return prompt
    
    def get_observation(self, response: dict) -> dict:
        """Execute the action and return the observation with confidence check."""
        # If no experiences were retrieved, use default agent logic (no confidence check)
        if not self._has_retrieved_experiences:
            return super().get_observation(response)
        
        # Otherwise, use experience retrieval logic with confidence check
        action = self.parse_action(response)
        output = self.execute_action(action)
        
        # Update workflow progress
        self.workflow_progress['steps'] += 1
        self._assess_workflow_progress(action['action'], output)
        
        # Check if this is a submission attempt
        action_str = action['action'].lower()
        if 'complete_task_and_submit_final_output' in action_str or 'git add' in action_str:
            # Validate submission
            if 'git diff --cached' in action_str or 'git diff' in action_str:
                # Check the diff output for issues
                diff_output = output.get('output', '')
                is_valid, reason = self._validate_fix_logic(diff_output)
                if not is_valid:
                    # Add warning to observation
                    warning = f"\n\n**Submission Validation Failed**: {reason}\nPlease check and fix the issue before resubmitting."
                    output['output'] = output.get('output', '') + warning
        
        # Render observation
        observation = self.render_template(self.config.action_observation_template, output=output)
        
        # Step 1: Check if we should extract confidence score from current response
        # The response content is already in messages (added in query()), so check current response
        response_content = response.get('content', '')
        confidence_score = self._extract_confidence_score(response_content)
        if confidence_score is not None:
            self.workflow_progress['confidence_score'] = confidence_score
            self.workflow_progress['last_confidence_check'] = self.workflow_progress['steps']
            
            # Step 2: If confidence score is high enough, prompt for submission
            # Skip confidence check in this step if we're prompting for submission
            submission_prompt = self._get_submission_prompt(confidence_score)
            if submission_prompt:
                observation = observation + "\n\n" + submission_prompt
                # Don't add confidence check prompt in the same observation
                # The LLM should focus on submission decision first
            else:
                # If score is low, still check if this is a key step for next confidence check
                if self._is_key_step(action['action'], output):
                    confidence_prompt = self._get_confidence_check_prompt()
                    if confidence_prompt:
                        observation = observation + "\n\n" + confidence_prompt
        else:
            # Step 3: No confidence score found, check if this is a key step
            # Ask LLM to assess its own confidence
            if self._is_key_step(action['action'], output):
                confidence_prompt = self._get_confidence_check_prompt()
                if confidence_prompt:
                    observation = observation + "\n\n" + confidence_prompt
        
        self.add_message("user", observation)
        return output
    
    def run(self, task: str, issue_id: Optional[str] = None, **kwargs) -> tuple[str, str]:
        """
        Run the agent with experience retrieval.
        If no experiences are found, completely fall back to DefaultAgent behavior.
        
        Args:
            task: The task description
            issue_id: Optional issue ID for targeted experience retrieval
            **kwargs: Additional keyword arguments
            
        Returns:
            Tuple of (exit_status, result)
        """
        # Reset workflow progress for new run
        self.workflow_progress = {
            'steps': 0,
            'has_code_changes': False,
            'test_passed': False,
            'last_confidence_check': None,
            'confidence_score': None
        }
        
        # Reset experience retrieval flag
        self._has_retrieved_experiences = False
        
        # Retrieve relevant experiences
        if self.experience_store:
            experiences = self.retrieve_experiences(task, issue_id)
            self._current_experiences = experiences
            
            # Log retrieval results for debugging
            if experiences:
                logger.info(f"Retrieved {len(experiences)} experience(s) for issue {issue_id}")
                for exp, sim in experiences:
                    logger.info(f"  - {exp.issue_id} (similarity: {sim:.4f})")
            else:
                logger.info(f"No experiences retrieved for issue {issue_id} (store_path: {self.experience_store.storage_path})")
            
            formatted_experiences = self.format_experiences(experiences)
            
            # Set flag based on whether experiences were retrieved
            self._has_retrieved_experiences = bool(formatted_experiences)
            
            # If no experiences found, completely fall back to DefaultAgent
            if not self._has_retrieved_experiences:
                # Clear any experience-related template vars
                self._current_experiences = []
                self.config.instance_template = self._original_instance_template
                # Remove experience-related vars from extra_template_vars
                self.extra_template_vars.pop("experiences", None)
                self.extra_template_vars.pop("formatted_experiences", None)
                # Call DefaultAgent.run() directly, bypassing all ExperienceRetrievalAgent logic
                return super().run(task, **kwargs)
            
            # Experiences found - proceed with ExperienceRetrievalAgent logic
            # Add experiences to extra_template_vars so they're available in templates
            self.extra_template_vars["experiences"] = experiences
            self.extra_template_vars["formatted_experiences"] = formatted_experiences
            
            # Insert experiences into instructions after "Recommended Workflow" section
            # Logical order: Overview -> Boundaries -> Workflow -> Experiences -> Execution Rules
            template = self._original_instance_template
            # Find end of "## Recommended Workflow" section (before next ##)
            if "## Recommended Workflow" in template and "\n    ## " in template:
                # Split at Recommended Workflow
                before_workflow, after_workflow = template.split("## Recommended Workflow", 1)
                # Find next section (## with proper indentation)
                if "\n    ## " in after_workflow:
                    workflow_content, rest = after_workflow.split("\n    ## ", 1)
                    # Insert experiences section between workflow and next section
                    # Use {% raw %} to prevent Jinja2 from parsing the already-rendered content
                    experiences_section = "\n\n    ## Relevant Experiences\n    {% raw %}\n    " + formatted_experiences.replace("\n", "\n    ") + "\n    {% endraw %}"
                    self.config.instance_template = (
                        before_workflow + 
                        "## Recommended Workflow" + 
                        workflow_content + 
                        experiences_section + 
                        "\n\n    ## " + 
                        rest
                    )
                else:
                    # No next section, insert before </instructions>
                    if "</instructions>" in after_workflow:
                        workflow_content, after_instructions = after_workflow.split("</instructions>", 1)
                        # Use {% raw %} to prevent Jinja2 from parsing the already-rendered content
                        experiences_section = "\n\n    ## Relevant Experiences\n    {% raw %}\n    " + formatted_experiences.replace("\n", "\n    ") + "\n    {% endraw %}"
                        self.config.instance_template = (
                            before_workflow + 
                            "## Recommended Workflow" + 
                            workflow_content + 
                            experiences_section + 
                            "\n    </instructions>" + 
                            after_instructions
                        )
                    else:
                        # Fallback - use {% raw %} to protect content
                        self.config.instance_template = "{% raw %}" + formatted_experiences + "{% endraw %}\n\n" + template
            else:
                # Fallback: prepend if structure not found - use {% raw %} to protect content
                self.config.instance_template = "{% raw %}" + formatted_experiences + "{% endraw %}\n\n" + template
        else:
            # No experience store - fall back to DefaultAgent
            self._current_experiences = []
            self._has_retrieved_experiences = False
            # Restore original template
            self.config.instance_template = self._original_instance_template
            # Clear experience-related vars
            self.extra_template_vars.pop("experiences", None)
            self.extra_template_vars.pop("formatted_experiences", None)
            # Call DefaultAgent.run() directly
            return super().run(task, **kwargs)
        
        # Call parent run method (only if experiences were found and used)
        try:
            return super().run(task, **kwargs)
        finally:
            # Restore original template after run completes
            self.config.instance_template = self._original_instance_template

