#!/usr/bin/env python3
"""
AI-Powered Workflow Recovery Agent
Uses AI to diagnose failures and generate automated fixes.
"""

import re
from typing import Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path

from .workflow_failure_analyzer import (
    WorkflowFailure,
    FailureCategory,
    RecoverySeverity
)


@dataclass
class RecoveryAction:
    """Represents a recovery action to be taken"""
    action_type: str
    description: str
    command: Optional[str] = None
    file_path: Optional[str] = None
    file_changes: Optional[str] = None
    confidence: float = 0.0
    requires_review: bool = True


class AIRecoveryAgent:
    """AI agent for automated workflow failure recovery"""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.recovery_templates = self._initialize_recovery_templates()
    
    def _initialize_recovery_templates(self) -> Dict[FailureCategory, List[RecoveryAction]]:
        """Initialize recovery action templates for each failure category"""
        return {
            FailureCategory.DEPENDENCY_ISSUE: [
                RecoveryAction(
                    action_type="update_requirements",
                    description="Update requirements.txt with missing package",
                    confidence=0.8,
                    requires_review=False
                ),
                RecoveryAction(
                    action_type="pin_version",
                    description="Pin package version to compatible version",
                    confidence=0.7,
                    requires_review=True
                ),
            ],
            
            FailureCategory.LINTING_ERROR: [
                RecoveryAction(
                    action_type="run_formatters",
                    description="Run code formatters automatically",
                    command="black src/ tests/ && isort src/ tests/ && ruff check --fix src/ tests/",
                    confidence=0.9,
                    requires_review=False
                ),
            ],
            
            FailureCategory.TIMEOUT: [
                RecoveryAction(
                    action_type="increase_timeout",
                    description="Increase workflow timeout",
                    confidence=0.8,
                    requires_review=True
                ),
            ],
            
            FailureCategory.NETWORK_ERROR: [
                RecoveryAction(
                    action_type="retry_workflow",
                    description="Retry workflow (transient network error)",
                    confidence=0.7,
                    requires_review=False
                ),
            ],
            
            FailureCategory.PERMISSION_ERROR: [
                RecoveryAction(
                    action_type="update_permissions",
                    description="Update workflow permissions",
                    confidence=0.6,
                    requires_review=True
                ),
            ],
        }
    
    def diagnose_and_recover(self, failure: WorkflowFailure) -> List[RecoveryAction]:
        """
        Diagnose a failure and generate recovery actions
        
        Args:
            failure: WorkflowFailure object to diagnose
        
        Returns:
            List of RecoveryAction objects
        """
        actions = []
        
        # Get template actions for this category
        template_actions = self.recovery_templates.get(failure.category, [])
        
        # Customize actions based on specific error details
        if failure.category == FailureCategory.DEPENDENCY_ISSUE:
            actions.extend(self._recover_dependency_issue(failure))
        
        elif failure.category == FailureCategory.LINTING_ERROR:
            actions.extend(self._recover_linting_error(failure))
        
        elif failure.category == FailureCategory.TIMEOUT:
            actions.extend(self._recover_timeout(failure))
        
        elif failure.category == FailureCategory.NETWORK_ERROR:
            actions.extend(self._recover_network_error(failure))
        
        elif failure.category == FailureCategory.PERMISSION_ERROR:
            actions.extend(self._recover_permission_error(failure))
        
        # If no specific recovery actions, use templates
        if not actions:
            actions = template_actions
        
        return actions
    
    def _recover_dependency_issue(self, failure: WorkflowFailure) -> List[RecoveryAction]:
        """Generate recovery actions for dependency issues"""
        actions = []
        
        # Extract package name from error message
        package_match = re.search(
            r"ModuleNotFoundError: No module named ['\"](.+?)['\"]",
            failure.error_message
        )
        
        if package_match:
            package_name = package_match.group(1).split('.')[0]
            
            actions.append(RecoveryAction(
                action_type="add_package",
                description=f"Add missing package '{package_name}' to requirements.txt",
                command=f"echo '{package_name}' >> requirements.txt",
                file_path="requirements.txt",
                confidence=0.8,
                requires_review=True
            ))
        
        # Check for version conflicts
        version_match = re.search(
            r"Could not find a version that satisfies.*?([a-zA-Z0-9_-]+)",
            failure.error_message
        )
        
        if version_match:
            package_name = version_match.group(1)
            
            actions.append(RecoveryAction(
                action_type="fix_version_conflict",
                description=f"Fix version conflict for '{package_name}'",
                confidence=0.6,
                requires_review=True
            ))
        
        return actions
    
    def _recover_linting_error(self, failure: WorkflowFailure) -> List[RecoveryAction]:
        """Generate recovery actions for linting errors"""
        actions = []
        
        # Determine which linter failed
        if "black" in failure.error_message.lower():
            actions.append(RecoveryAction(
                action_type="run_black",
                description="Run Black formatter",
                command="black src/ tests/",
                confidence=0.95,
                requires_review=False
            ))
        
        if "isort" in failure.error_message.lower():
            actions.append(RecoveryAction(
                action_type="run_isort",
                description="Run isort import sorter",
                command="isort src/ tests/",
                confidence=0.95,
                requires_review=False
            ))
        
        if "ruff" in failure.error_message.lower():
            actions.append(RecoveryAction(
                action_type="run_ruff",
                description="Run Ruff auto-fixer",
                command="ruff check --fix src/ tests/",
                confidence=0.9,
                requires_review=False
            ))
        
        if "flake8" in failure.error_message.lower():
            actions.append(RecoveryAction(
                action_type="run_flake8_fix",
                description="Fix flake8 issues (may need manual review)",
                confidence=0.7,
                requires_review=True
            ))
        
        return actions
    
    def _recover_timeout(self, failure: WorkflowFailure) -> List[RecoveryAction]:
        """Generate recovery actions for timeout issues"""
        actions = []
        
        # Check if it's a workflow timeout
        if "workflow" in failure.step_name.lower():
            actions.append(RecoveryAction(
                action_type="increase_workflow_timeout",
                description="Increase workflow timeout in .github/workflows/*.yml",
                confidence=0.8,
                requires_review=True
            ))
        
        # Check if it's a step timeout
        else:
            actions.append(RecoveryAction(
                action_type="increase_step_timeout",
                description=f"Increase timeout for step '{failure.step_name}'",
                confidence=0.8,
                requires_review=True
            ))
        
        return actions
    
    def _recover_network_error(self, failure: WorkflowFailure) -> List[RecoveryAction]:
        """Generate recovery actions for network errors"""
        actions = []
        
        # Network errors are often transient
        actions.append(RecoveryAction(
            action_type="retry",
            description="Retry workflow (transient network error)",
            confidence=0.7,
            requires_review=False
        ))
        
        # Check if it's a specific service failure
        if "pypi" in failure.error_message.lower() or "pip" in failure.error_message.lower():
            actions.append(RecoveryAction(
                action_type="use_cache",
                description="Enable pip caching in workflow",
                confidence=0.6,
                requires_review=True
            ))
        
        return actions
    
    def _recover_permission_error(self, failure: WorkflowFailure) -> List[RecoveryAction]:
        """Generate recovery actions for permission errors"""
        actions = []
        
        # Check for workflow permission issues
        if "security-events" in failure.error_message or "GITHUB_TOKEN" in failure.error_message:
            actions.append(RecoveryAction(
                action_type="update_workflow_permissions",
                description="Update workflow permissions in .github/workflows/*.yml",
                file_changes="""
Add or update permissions section:
permissions:
  security-events: write
  actions: read
  contents: read
                """,
                confidence=0.7,
                requires_review=True
            ))
        
        return actions
    
    def generate_fix_pr_description(
        self,
        failure: WorkflowFailure,
        actions: List[RecoveryAction]
    ) -> str:
        """Generate PR description for automated fix"""
        description = f"""## 🤖 Automated Workflow Failure Recovery

### Failure Details
- **Workflow:** {failure.workflow_name}
- **Job:** {failure.job_name}
- **Step:** {failure.step_name}
- **Category:** {failure.category.value}
- **Severity:** {failure.severity.value}
- **Timestamp:** {failure.timestamp.isoformat()}

### Error Message
```
{failure.error_message}
```

### Recovery Actions Taken
"""
        
        for idx, action in enumerate(actions, 1):
            description += f"\n{idx}. **{action.description}**\n"
            description += f"   - Type: `{action.action_type}`\n"
            description += f"   - Confidence: {action.confidence * 100:.0f}%\n"
            
            if action.command:
                description += f"   - Command: `{action.command}`\n"
            
            if action.file_path:
                description += f"   - File: `{action.file_path}`\n"
        
        description += f"""
### Log Excerpt
```
{failure.log_excerpt}
```

### Review Required
{"⚠️ This PR requires manual review before merging." if any(a.requires_review for a in actions) else "✅ This PR can be auto-merged if CI passes."}

---
*This PR was automatically generated by the AI Recovery Agent*
"""
        
        return description
    
    def can_auto_recover(self, failure: WorkflowFailure) -> bool:
        """Determine if a failure can be automatically recovered"""
        if not failure.auto_fixable:
            return False
        
        if failure.severity in [RecoverySeverity.HIGH, RecoverySeverity.CRITICAL]:
            return False
        
        # Only auto-recover specific categories
        auto_recoverable_categories = [
            FailureCategory.LINTING_ERROR,
            FailureCategory.NETWORK_ERROR,
        ]
        
        return failure.category in auto_recoverable_categories
    
    def execute_recovery_actions(
        self,
        actions: List[RecoveryAction],
        dry_run: bool = True
    ) -> Dict[str, any]:
        """
        Execute recovery actions
        
        Args:
            actions: List of recovery actions to execute
            dry_run: If True, only simulate actions without executing
        
        Returns:
            Dictionary with execution results
        """
        results = {
            "success": False,
            "actions_executed": [],
            "actions_failed": [],
            "changes_made": [],
        }
        
        for action in actions:
            try:
                if action.command and not dry_run:
                    # Execute command
                    import subprocess
                    result = subprocess.run(
                        action.command,
                        shell=True,
                        capture_output=True,
                        text=True,
                        cwd=str(self.repo_path)
                    )
                    
                    if result.returncode == 0:
                        results["actions_executed"].append({
                            "action": action.action_type,
                            "description": action.description,
                            "output": result.stdout
                        })
                        results["changes_made"].append(action.file_path or "command executed")
                    else:
                        results["actions_failed"].append({
                            "action": action.action_type,
                            "error": result.stderr
                        })
                else:
                    # Dry run or no command
                    results["actions_executed"].append({
                        "action": action.action_type,
                        "description": action.description,
                        "dry_run": dry_run
                    })
            
            except Exception as e:
                results["actions_failed"].append({
                    "action": action.action_type,
                    "error": str(e)
                })
        
        results["success"] = len(results["actions_failed"]) == 0
        
        return results
    
    def generate_recovery_script(self, actions: List[RecoveryAction]) -> str:
        """Generate a shell script to execute recovery actions"""
        script = """#!/bin/bash
# Auto-generated recovery script
set -e

echo "🤖 Starting automated recovery..."

"""
        
        for action in actions:
            script += f"\n# {action.description}\n"
            if action.command:
                script += f"{action.command}\n"
            else:
                script += f"# Manual action required: {action.action_type}\n"
        
        script += """
echo "✅ Recovery completed!"
"""
        
        return script
