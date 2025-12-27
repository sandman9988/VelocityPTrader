#!/usr/bin/env python3
"""
Workflow Failure Analyzer
Analyzes GitHub Actions workflow failures and categorizes them for auto-recovery.
"""

import re
from typing import Dict, List, Optional
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timezone


class FailureCategory(Enum):
    """Categories of workflow failures"""
    DEPENDENCY_ISSUE = "dependency_issue"
    SYNTAX_ERROR = "syntax_error"
    TEST_FAILURE = "test_failure"
    LINTING_ERROR = "linting_error"
    TYPE_ERROR = "type_error"
    SECURITY_SCAN = "security_scan"
    BUILD_ERROR = "build_error"
    TIMEOUT = "timeout"
    NETWORK_ERROR = "network_error"
    PERMISSION_ERROR = "permission_error"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    CONFIGURATION_ERROR = "configuration_error"
    UNKNOWN = "unknown"


class RecoverySeverity(Enum):
    """Severity levels for recovery actions"""
    LOW = "low"  # Can be auto-fixed
    MEDIUM = "medium"  # Can be attempted with confidence
    HIGH = "high"  # Requires careful review
    CRITICAL = "critical"  # Manual intervention required


@dataclass
class FailurePattern:
    """Pattern for identifying specific failure types"""
    category: FailureCategory
    severity: RecoverySeverity
    patterns: List[str]
    description: str
    auto_fixable: bool = False
    fix_template: Optional[str] = None


@dataclass
class WorkflowFailure:
    """Represents a workflow failure"""
    workflow_name: str
    job_name: str
    step_name: str
    error_message: str
    log_excerpt: str
    category: FailureCategory
    severity: RecoverySeverity
    timestamp: datetime
    auto_fixable: bool
    suggested_fix: Optional[str] = None
    patterns_matched: List[str] = field(default_factory=list)


class WorkflowFailureAnalyzer:
    """Analyzes workflow failures and suggests recovery actions"""
    
    def __init__(self):
        self.failure_patterns = self._initialize_patterns()
    
    def _initialize_patterns(self) -> List[FailurePattern]:
        """Initialize common failure patterns"""
        return [
            # Dependency issues
            FailurePattern(
                category=FailureCategory.DEPENDENCY_ISSUE,
                severity=RecoverySeverity.LOW,
                patterns=[
                    r"ModuleNotFoundError: No module named '(.+)'",
                    r"ImportError: cannot import name '(.+)'",
                    r"pip.*ERROR.*No matching distribution found",
                    r"Could not find a version that satisfies",
                    r"ERROR: Could not install packages",
                ],
                description="Missing or incompatible dependencies",
                auto_fixable=True,
                fix_template="Add missing package to requirements.txt or update version constraints"
            ),
            
            # Syntax errors
            FailurePattern(
                category=FailureCategory.SYNTAX_ERROR,
                severity=RecoverySeverity.MEDIUM,
                patterns=[
                    r"SyntaxError: (.+)",
                    r"IndentationError: (.+)",
                    r"TabError: (.+)",
                ],
                description="Python syntax errors",
                auto_fixable=False,
                fix_template="Fix syntax error in source code"
            ),
            
            # Test failures
            FailurePattern(
                category=FailureCategory.TEST_FAILURE,
                severity=RecoverySeverity.MEDIUM,
                patterns=[
                    r"FAILED tests/(.+)::(.+)",
                    r"AssertionError: (.+)",
                    r"pytest.*failed.*passed",
                    r"test.*FAILED",
                ],
                description="Unit or integration test failures",
                auto_fixable=False,
                fix_template="Investigate and fix failing tests"
            ),
            
            # Linting errors
            FailurePattern(
                category=FailureCategory.LINTING_ERROR,
                severity=RecoverySeverity.LOW,
                patterns=[
                    r"ruff.*error",
                    r"flake8.*error",
                    r"pylint.*error",
                    r"black.*would reformat",
                    r"isort.*ERROR",
                ],
                description="Code style or linting violations",
                auto_fixable=True,
                fix_template="Run formatters: black, isort, ruff --fix"
            ),
            
            # Type errors
            FailurePattern(
                category=FailureCategory.TYPE_ERROR,
                severity=RecoverySeverity.MEDIUM,
                patterns=[
                    r"mypy.*error:",
                    r"TypeError: (.+)",
                    r"AttributeError: (.+)",
                ],
                description="Type checking errors",
                auto_fixable=False,
                fix_template="Fix type annotations or type mismatches"
            ),
            
            # Security scan issues
            FailurePattern(
                category=FailureCategory.SECURITY_SCAN,
                severity=RecoverySeverity.HIGH,
                patterns=[
                    r"bandit.*Issue: \[(.+)\]",
                    r"safety.*vulnerability",
                    r"semgrep.*finding",
                    r"CodeQL.*alert",
                ],
                description="Security vulnerabilities detected",
                auto_fixable=False,
                fix_template="Address security vulnerabilities"
            ),
            
            # Build errors
            FailurePattern(
                category=FailureCategory.BUILD_ERROR,
                severity=RecoverySeverity.MEDIUM,
                patterns=[
                    r"build.*failed",
                    r"compilation error",
                    r"Error: Process completed with exit code (\d+)",
                ],
                description="Build process failures",
                auto_fixable=False,
                fix_template="Fix build configuration or dependencies"
            ),
            
            # Timeouts
            FailurePattern(
                category=FailureCategory.TIMEOUT,
                severity=RecoverySeverity.LOW,
                patterns=[
                    r"timeout",
                    r"Timeout waiting for",
                    r"Operation timed out",
                    r"exceeded.*time limit",
                ],
                description="Operation timeout",
                auto_fixable=True,
                fix_template="Increase timeout or optimize operation"
            ),
            
            # Network errors
            FailurePattern(
                category=FailureCategory.NETWORK_ERROR,
                severity=RecoverySeverity.LOW,
                patterns=[
                    r"ConnectionError",
                    r"Network is unreachable",
                    r"Failed to connect",
                    r"Could not resolve host",
                ],
                description="Network connectivity issues",
                auto_fixable=True,
                fix_template="Retry operation or check network configuration"
            ),
            
            # Permission errors
            FailurePattern(
                category=FailureCategory.PERMISSION_ERROR,
                severity=RecoverySeverity.MEDIUM,
                patterns=[
                    r"PermissionError",
                    r"Permission denied",
                    r"Access denied",
                    r"Forbidden",
                ],
                description="File or resource permission issues",
                auto_fixable=True,
                fix_template="Check file permissions or workflow permissions"
            ),
            
            # Resource exhaustion
            FailurePattern(
                category=FailureCategory.RESOURCE_EXHAUSTION,
                severity=RecoverySeverity.MEDIUM,
                patterns=[
                    r"Out of memory",
                    r"Disk space",
                    r"No space left",
                    r"MemoryError",
                ],
                description="Resource exhaustion (memory, disk, etc.)",
                auto_fixable=True,
                fix_template="Increase resources or optimize usage"
            ),
            
            # Configuration errors
            FailurePattern(
                category=FailureCategory.CONFIGURATION_ERROR,
                severity=RecoverySeverity.MEDIUM,
                patterns=[
                    r"configuration.*invalid",
                    r"ConfigError",
                    r"Missing required.*config",
                    r"Invalid configuration",
                ],
                description="Configuration file errors",
                auto_fixable=False,
                fix_template="Fix configuration file"
            ),
        ]
    
    def analyze_failure(
        self,
        workflow_name: str,
        job_name: str,
        step_name: str,
        log_content: str,
        timestamp: Optional[datetime] = None
    ) -> WorkflowFailure:
        """
        Analyze a workflow failure and categorize it
        
        Args:
            workflow_name: Name of the workflow
            job_name: Name of the failed job
            step_name: Name of the failed step
            log_content: Log content from the failed step
            timestamp: Timestamp of the failure
        
        Returns:
            WorkflowFailure object with analysis results
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        # Extract error message
        error_message = self._extract_error_message(log_content)
        
        # Match against patterns
        category = FailureCategory.UNKNOWN
        severity = RecoverySeverity.CRITICAL
        auto_fixable = False
        suggested_fix = None
        patterns_matched = []
        
        for pattern_def in self.failure_patterns:
            for pattern in pattern_def.patterns:
                if re.search(pattern, log_content, re.IGNORECASE):
                    category = pattern_def.category
                    severity = pattern_def.severity
                    auto_fixable = pattern_def.auto_fixable
                    suggested_fix = pattern_def.fix_template
                    patterns_matched.append(pattern)
                    break
            if category != FailureCategory.UNKNOWN:
                break
        
        # Extract relevant log excerpt
        log_excerpt = self._extract_relevant_excerpt(log_content, error_message)
        
        return WorkflowFailure(
            workflow_name=workflow_name,
            job_name=job_name,
            step_name=step_name,
            error_message=error_message,
            log_excerpt=log_excerpt,
            category=category,
            severity=severity,
            timestamp=timestamp,
            auto_fixable=auto_fixable,
            suggested_fix=suggested_fix,
            patterns_matched=patterns_matched
        )
    
    def _extract_error_message(self, log_content: str) -> str:
        """Extract the primary error message from log content"""
        error_patterns = [
            r"Error: (.+?)(?:\n|$)",
            r"ERROR: (.+?)(?:\n|$)",
            r"FAILED (.+?)(?:\n|$)",
            r"Exception: (.+?)(?:\n|$)",
            r"(?:^|\n)(.+?error.+?)(?:\n|$)",
        ]
        
        for pattern in error_patterns:
            match = re.search(pattern, log_content, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).strip()
        
        # If no specific error pattern found, return first non-empty line
        lines = [line.strip() for line in log_content.split('\n') if line.strip()]
        return lines[0] if lines else "Unknown error"
    
    def _extract_relevant_excerpt(self, log_content: str, error_message: str, context_lines: int = 5) -> str:
        """Extract relevant log excerpt around the error"""
        lines = log_content.split('\n')
        
        # Find the line with the error message
        error_line_idx = -1
        for idx, line in enumerate(lines):
            if error_message in line:
                error_line_idx = idx
                break
        
        if error_line_idx == -1:
            # If error message not found, return last N lines
            return '\n'.join(lines[-context_lines:])
        
        # Extract context around error
        start_idx = max(0, error_line_idx - context_lines)
        end_idx = min(len(lines), error_line_idx + context_lines + 1)
        
        return '\n'.join(lines[start_idx:end_idx])
    
    def generate_recovery_report(self, failure: WorkflowFailure) -> Dict:
        """Generate a structured recovery report"""
        return {
            "workflow": failure.workflow_name,
            "job": failure.job_name,
            "step": failure.step_name,
            "timestamp": failure.timestamp.isoformat(),
            "category": failure.category.value,
            "severity": failure.severity.value,
            "auto_fixable": failure.auto_fixable,
            "error_message": failure.error_message,
            "suggested_fix": failure.suggested_fix,
            "patterns_matched": failure.patterns_matched,
            "log_excerpt": failure.log_excerpt,
            "recovery_actions": self._generate_recovery_actions(failure)
        }
    
    def _generate_recovery_actions(self, failure: WorkflowFailure) -> List[str]:
        """Generate specific recovery actions based on failure category"""
        actions = []
        
        if failure.category == FailureCategory.DEPENDENCY_ISSUE:
            actions.extend([
                "Check requirements.txt for missing or incompatible packages",
                "Verify package versions compatibility",
                "Run 'pip install -r requirements.txt' locally to test",
                "Consider adding version constraints if needed"
            ])
        
        elif failure.category == FailureCategory.LINTING_ERROR:
            actions.extend([
                "Run 'black src/ tests/' to format code",
                "Run 'isort src/ tests/' to sort imports",
                "Run 'ruff check --fix src/ tests/' to auto-fix issues",
                "Review and commit formatting changes"
            ])
        
        elif failure.category == FailureCategory.TEST_FAILURE:
            actions.extend([
                "Run failed tests locally to reproduce",
                "Check test assertions and expected values",
                "Verify test data and fixtures",
                "Update tests if requirements changed"
            ])
        
        elif failure.category == FailureCategory.TIMEOUT:
            actions.extend([
                "Increase timeout value in workflow configuration",
                "Optimize slow operations",
                "Consider splitting long-running jobs",
                "Check for infinite loops or deadlocks"
            ])
        
        elif failure.category == FailureCategory.NETWORK_ERROR:
            actions.extend([
                "Retry the workflow",
                "Check external service status",
                "Add retry logic to network operations",
                "Consider using cached dependencies"
            ])
        
        else:
            actions.append("Manual investigation required")
        
        return actions
    
    def batch_analyze_failures(self, failures_data: List[Dict]) -> List[WorkflowFailure]:
        """Analyze multiple failures in batch"""
        results = []
        for failure_data in failures_data:
            failure = self.analyze_failure(
                workflow_name=failure_data.get('workflow', 'Unknown'),
                job_name=failure_data.get('job', 'Unknown'),
                step_name=failure_data.get('step', 'Unknown'),
                log_content=failure_data.get('log', ''),
                timestamp=failure_data.get('timestamp')
            )
            results.append(failure)
        return results
    
    def get_failure_statistics(self, failures: List[WorkflowFailure]) -> Dict:
        """Generate statistics from analyzed failures"""
        if not failures:
            return {
                "total_failures": 0,
                "by_category": {},
                "by_severity": {},
                "auto_fixable_count": 0,
                "auto_fixable_percentage": 0.0
            }
        
        stats = {
            "total_failures": len(failures),
            "by_category": {},
            "by_severity": {},
            "auto_fixable_count": sum(1 for f in failures if f.auto_fixable),
            "auto_fixable_percentage": 0.0
        }
        
        for failure in failures:
            # Count by category
            cat = failure.category.value
            stats["by_category"][cat] = stats["by_category"].get(cat, 0) + 1
            
            # Count by severity
            sev = failure.severity.value
            stats["by_severity"][sev] = stats["by_severity"].get(sev, 0) + 1
        
        stats["auto_fixable_percentage"] = (
            stats["auto_fixable_count"] / stats["total_failures"] * 100
        )
        
        return stats
