#!/usr/bin/env python3
"""
Tests for Workflow Failure Analyzer
"""

import pytest
from datetime import datetime
from pathlib import Path
import sys

import sys
from pathlib import Path
from datetime import datetime

# Add src to path for test execution
# This allows tests to run without package installation
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.workflow_failure_analyzer import (
    WorkflowFailureAnalyzer,
    FailureCategory,
    RecoverySeverity,
    FailurePattern
)


class TestWorkflowFailureAnalyzer:
    """Test suite for WorkflowFailureAnalyzer"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.analyzer = WorkflowFailureAnalyzer()
    
    def test_initialization(self):
        """Test analyzer initialization"""
        assert self.analyzer is not None
        assert len(self.analyzer.failure_patterns) > 0
    
    def test_dependency_issue_detection(self):
        """Test detection of dependency issues"""
        log_content = """
        Running pip install...
        ERROR: Could not find a version that satisfies the requirement pandas
        ModuleNotFoundError: No module named 'numpy'
        """
        
        failure = self.analyzer.analyze_failure(
            workflow_name="CI/CD",
            job_name="test",
            step_name="Install dependencies",
            log_content=log_content
        )
        
        assert failure.category == FailureCategory.DEPENDENCY_ISSUE
        assert failure.auto_fixable == True
        assert failure.severity == RecoverySeverity.LOW
    
    def test_linting_error_detection(self):
        """Test detection of linting errors"""
        log_content = """
        Running black...
        black: error: would reformat src/main.py
        1 file would be reformatted
        """
        
        failure = self.analyzer.analyze_failure(
            workflow_name="CI/CD",
            job_name="lint",
            step_name="Run Black",
            log_content=log_content
        )
        
        assert failure.category == FailureCategory.LINTING_ERROR
        assert failure.auto_fixable == True
    
    def test_syntax_error_detection(self):
        """Test detection of syntax errors"""
        log_content = """
        File "src/main.py", line 42
            def invalid syntax:
                            ^
        SyntaxError: invalid syntax
        """
        
        failure = self.analyzer.analyze_failure(
            workflow_name="CI/CD",
            job_name="test",
            step_name="Run tests",
            log_content=log_content
        )
        
        assert failure.category == FailureCategory.SYNTAX_ERROR
        assert failure.auto_fixable == False
        assert failure.severity == RecoverySeverity.MEDIUM
    
    def test_test_failure_detection(self):
        """Test detection of test failures"""
        log_content = """
        tests/test_main.py::test_feature FAILED
        AssertionError: Expected 5, got 3
        """
        
        failure = self.analyzer.analyze_failure(
            workflow_name="CI/CD",
            job_name="test",
            step_name="Run pytest",
            log_content=log_content
        )
        
        assert failure.category == FailureCategory.TEST_FAILURE
        assert failure.auto_fixable == False
    
    def test_timeout_detection(self):
        """Test detection of timeout issues"""
        log_content = """
        timeout: Operation timed out after 300 seconds
        """
        
        failure = self.analyzer.analyze_failure(
            workflow_name="CI/CD",
            job_name="build",
            step_name="Build project",
            log_content=log_content
        )
        
        assert failure.category == FailureCategory.TIMEOUT
        assert failure.auto_fixable == True
    
    def test_network_error_detection(self):
        """Test detection of network errors"""
        log_content = """
        ConnectionError: Failed to download from https://pypi.org
        Network is unreachable
        """
        
        failure = self.analyzer.analyze_failure(
            workflow_name="CI/CD",
            job_name="setup",
            step_name="Install packages",
            log_content=log_content
        )
        
        assert failure.category == FailureCategory.NETWORK_ERROR
        assert failure.auto_fixable == True
    
    def test_error_message_extraction(self):
        """Test error message extraction"""
        log_content = """
        Some log output
        Error: This is the main error message
        More log output
        """
        
        error_msg = self.analyzer._extract_error_message(log_content)
        assert "This is the main error message" in error_msg
    
    def test_log_excerpt_extraction(self):
        """Test relevant log excerpt extraction"""
        log_lines = [f"Line {i}" for i in range(20)]
        log_content = '\n'.join(log_lines)
        error_message = "Line 10"
        
        excerpt = self.analyzer._extract_relevant_excerpt(log_content, error_message, context_lines=2)
        
        assert "Line 8" in excerpt
        assert "Line 10" in excerpt
        assert "Line 12" in excerpt
    
    def test_generate_recovery_report(self):
        """Test recovery report generation"""
        log_content = "black would reformat file.py"
        
        failure = self.analyzer.analyze_failure(
            workflow_name="CI/CD",
            job_name="lint",
            step_name="Run Black",
            log_content=log_content
        )
        
        report = self.analyzer.generate_recovery_report(failure)
        
        assert "workflow" in report
        assert "category" in report
        assert "severity" in report
        assert "auto_fixable" in report
        assert "recovery_actions" in report
        assert len(report["recovery_actions"]) > 0
    
    def test_batch_analyze_failures(self):
        """Test batch failure analysis"""
        failures_data = [
            {
                "workflow": "CI/CD",
                "job": "lint",
                "step": "Black",
                "log": "black would reformat file.py"
            },
            {
                "workflow": "CI/CD",
                "job": "test",
                "step": "pytest",
                "log": "FAILED tests/test_main.py::test_feature"
            }
        ]
        
        results = self.analyzer.batch_analyze_failures(failures_data)
        
        assert len(results) == 2
        assert results[0].category == FailureCategory.LINTING_ERROR
        assert results[1].category == FailureCategory.TEST_FAILURE
    
    def test_failure_statistics(self):
        """Test failure statistics generation"""
        failures_data = [
            {
                "workflow": "CI/CD",
                "job": "lint",
                "step": "Black",
                "log": "black: error would reformat file.py"
            },
            {
                "workflow": "CI/CD",
                "job": "lint",
                "step": "isort",
                "log": "isort: ERROR would reformat imports"
            },
            {
                "workflow": "CI/CD",
                "job": "test",
                "step": "pytest",
                "log": "FAILED tests/test_main.py::test_feature - AssertionError"
            }
        ]
        
        failures = self.analyzer.batch_analyze_failures(failures_data)
        stats = self.analyzer.get_failure_statistics(failures)
        
        assert stats["total_failures"] == 3
        assert stats["auto_fixable_count"] == 2
        assert stats["auto_fixable_percentage"] > 0
        assert "linting_error" in stats["by_category"]
        assert "test_failure" in stats["by_category"]
    
    def test_unknown_failure_category(self):
        """Test handling of unknown failure types"""
        log_content = "Some random error that doesn't match any pattern"
        
        failure = self.analyzer.analyze_failure(
            workflow_name="CI/CD",
            job_name="unknown",
            step_name="Unknown step",
            log_content=log_content
        )
        
        assert failure.category == FailureCategory.UNKNOWN
        assert failure.severity == RecoverySeverity.CRITICAL
        assert failure.auto_fixable == False
    
    def test_recovery_actions_for_linting(self):
        """Test specific recovery actions for linting errors"""
        log_content = "black would reformat file.py"
        
        failure = self.analyzer.analyze_failure(
            workflow_name="CI/CD",
            job_name="lint",
            step_name="Run Black",
            log_content=log_content
        )
        
        actions = self.analyzer._generate_recovery_actions(failure)
        
        assert len(actions) > 0
        assert any("black" in action.lower() for action in actions)
    
    def test_recovery_actions_for_tests(self):
        """Test specific recovery actions for test failures"""
        log_content = "FAILED tests/test_main.py::test_feature"
        
        failure = self.analyzer.analyze_failure(
            workflow_name="CI/CD",
            job_name="test",
            step_name="Run tests",
            log_content=log_content
        )
        
        actions = self.analyzer._generate_recovery_actions(failure)
        
        assert len(actions) > 0
        assert any("test" in action.lower() for action in actions)
    
    def test_timestamp_handling(self):
        """Test timestamp handling in failure analysis"""
        log_content = "Error occurred"
        custom_timestamp = datetime(2024, 1, 1, 12, 0, 0)
        
        failure = self.analyzer.analyze_failure(
            workflow_name="CI/CD",
            job_name="test",
            step_name="Run",
            log_content=log_content,
            timestamp=custom_timestamp
        )
        
        assert failure.timestamp == custom_timestamp


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
