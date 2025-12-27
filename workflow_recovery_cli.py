#!/usr/bin/env python3
"""
Workflow Failure Recovery CLI
Command-line tool for analyzing and recovering from workflow failures.
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path for local development and testing
# This allows the CLI to work without package installation
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.workflow_failure_analyzer import WorkflowFailureAnalyzer
from src.utils.ai_recovery_agent import AIRecoveryAgent


def analyze_log_file(log_file: Path, workflow_name: str, job_name: str, step_name: str):
    """Analyze a single log file"""
    analyzer = WorkflowFailureAnalyzer()
    
    log_content = log_file.read_text()
    
    failure = analyzer.analyze_failure(
        workflow_name=workflow_name,
        job_name=job_name,
        step_name=step_name,
        log_content=log_content
    )
    
    return failure


def print_failure_report(failure):
    """Print formatted failure report"""
    print("\n" + "=" * 80)
    print("WORKFLOW FAILURE ANALYSIS REPORT")
    print("=" * 80)
    print(f"\n📋 Workflow: {failure.workflow_name}")
    print(f"🔧 Job: {failure.job_name}")
    print(f"⚙️  Step: {failure.step_name}")
    print(f"📅 Timestamp: {failure.timestamp}")
    print(f"\n🏷️  Category: {failure.category.value}")
    print(f"⚠️  Severity: {failure.severity.value}")
    print(f"🤖 Auto-fixable: {'Yes' if failure.auto_fixable else 'No'}")
    
    print(f"\n❌ Error Message:")
    print(f"   {failure.error_message}")
    
    if failure.suggested_fix:
        print(f"\n💡 Suggested Fix:")
        print(f"   {failure.suggested_fix}")
    
    print(f"\n📝 Log Excerpt:")
    print("-" * 80)
    for line in failure.log_excerpt.split('\n'):
        print(f"   {line}")
    print("-" * 80)
    
    if failure.patterns_matched:
        print(f"\n🔍 Patterns Matched:")
        for pattern in failure.patterns_matched:
            print(f"   - {pattern}")


def print_recovery_actions(actions):
    """Print recovery actions"""
    print("\n" + "=" * 80)
    print("RECOVERY ACTIONS")
    print("=" * 80)
    
    if not actions:
        print("\n⚠️  No specific recovery actions available")
        return
    
    for idx, action in enumerate(actions, 1):
        print(f"\n{idx}. {action.description}")
        print(f"   Type: {action.action_type}")
        print(f"   Confidence: {action.confidence * 100:.0f}%")
        print(f"   Requires Review: {'Yes' if action.requires_review else 'No'}")
        
        if action.command:
            print(f"   Command: {action.command}")
        
        if action.file_path:
            print(f"   File: {action.file_path}")
        
        if action.file_changes:
            print(f"   Changes:")
            for line in action.file_changes.strip().split('\n'):
                print(f"      {line}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and recover from workflow failures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a log file
  python workflow_recovery_cli.py analyze --log workflow.log --workflow "CI/CD" --job "test" --step "Run tests"
  
  # Analyze and generate recovery actions
  python workflow_recovery_cli.py recover --log workflow.log --workflow "CI/CD" --job "test" --step "Run tests"
  
  # Execute recovery actions
  python workflow_recovery_cli.py execute --log workflow.log --workflow "CI/CD" --job "test" --step "Run tests"
  
  # Analyze directory of logs
  python workflow_recovery_cli.py batch --logs-dir ./workflow-logs --workflow "CI/CD"
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze workflow failure')
    analyze_parser.add_argument('--log', type=Path, required=True, help='Path to log file')
    analyze_parser.add_argument('--workflow', required=True, help='Workflow name')
    analyze_parser.add_argument('--job', required=True, help='Job name')
    analyze_parser.add_argument('--step', required=True, help='Step name')
    analyze_parser.add_argument('--output', type=Path, help='Output JSON file')
    
    # Recover command
    recover_parser = subparsers.add_parser('recover', help='Analyze and generate recovery actions')
    recover_parser.add_argument('--log', type=Path, required=True, help='Path to log file')
    recover_parser.add_argument('--workflow', required=True, help='Workflow name')
    recover_parser.add_argument('--job', required=True, help='Job name')
    recover_parser.add_argument('--step', required=True, help='Step name')
    recover_parser.add_argument('--output', type=Path, help='Output JSON file')
    
    # Execute command
    execute_parser = subparsers.add_parser('execute', help='Execute recovery actions')
    execute_parser.add_argument('--log', type=Path, required=True, help='Path to log file')
    execute_parser.add_argument('--workflow', required=True, help='Workflow name')
    execute_parser.add_argument('--job', required=True, help='Job name')
    execute_parser.add_argument('--step', required=True, help='Step name')
    execute_parser.add_argument('--dry-run', action='store_true', help='Simulate execution without making changes')
    
    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Analyze multiple log files')
    batch_parser.add_argument('--logs-dir', type=Path, required=True, help='Directory containing log files')
    batch_parser.add_argument('--workflow', required=True, help='Workflow name')
    batch_parser.add_argument('--output', type=Path, help='Output JSON file')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        if args.command == 'analyze':
            failure = analyze_log_file(args.log, args.workflow, args.job, args.step)
            print_failure_report(failure)
            
            if args.output:
                analyzer = WorkflowFailureAnalyzer()
                report = analyzer.generate_recovery_report(failure)
                args.output.write_text(json.dumps(report, indent=2))
                print(f"\n✅ Report saved to {args.output}")
        
        elif args.command == 'recover':
            failure = analyze_log_file(args.log, args.workflow, args.job, args.step)
            print_failure_report(failure)
            
            agent = AIRecoveryAgent()
            actions = agent.diagnose_and_recover(failure)
            print_recovery_actions(actions)
            
            if args.output:
                output_data = {
                    "failure": {
                        "workflow": failure.workflow_name,
                        "job": failure.job_name,
                        "step": failure.step_name,
                        "category": failure.category.value,
                        "severity": failure.severity.value,
                        "auto_fixable": failure.auto_fixable,
                        "error_message": failure.error_message,
                    },
                    "recovery_actions": [
                        {
                            "type": a.action_type,
                            "description": a.description,
                            "command": a.command,
                            "confidence": a.confidence,
                            "requires_review": a.requires_review,
                        }
                        for a in actions
                    ]
                }
                args.output.write_text(json.dumps(output_data, indent=2))
                print(f"\n✅ Recovery plan saved to {args.output}")
        
        elif args.command == 'execute':
            failure = analyze_log_file(args.log, args.workflow, args.job, args.step)
            agent = AIRecoveryAgent()
            actions = agent.diagnose_and_recover(failure)
            
            print_failure_report(failure)
            print_recovery_actions(actions)
            
            if args.dry_run:
                print("\n🔍 DRY RUN MODE - No changes will be made")
            
            print("\n🚀 Executing recovery actions...")
            results = agent.execute_recovery_actions(actions, dry_run=args.dry_run)
            
            print("\n" + "=" * 80)
            print("EXECUTION RESULTS")
            print("=" * 80)
            
            if results["success"]:
                print("\n✅ All actions executed successfully")
            else:
                print("\n❌ Some actions failed")
            
            if results["actions_executed"]:
                print("\n📋 Actions Executed:")
                for action in results["actions_executed"]:
                    print(f"   ✅ {action['description']}")
            
            if results["actions_failed"]:
                print("\n❌ Actions Failed:")
                for action in results["actions_failed"]:
                    print(f"   ❌ {action['action']}: {action['error']}")
            
            if results["changes_made"]:
                print("\n📝 Changes Made:")
                for change in results["changes_made"]:
                    print(f"   - {change}")
        
        elif args.command == 'batch':
            analyzer = WorkflowFailureAnalyzer()
            failures = []
            
            print(f"🔍 Scanning {args.logs_dir} for log files...")
            
            for log_file in args.logs_dir.rglob("*.txt"):
                try:
                    parts = log_file.parts
                    job_name = parts[-2] if len(parts) > 1 else "unknown"
                    step_name = log_file.stem
                    
                    failure = analyze_log_file(log_file, args.workflow, job_name, step_name)
                    failures.append(failure)
                    
                    print(f"   ✅ Analyzed: {log_file.name}")
                except Exception as e:
                    print(f"   ❌ Error analyzing {log_file.name}: {e}")
            
            if failures:
                print(f"\n📊 Analyzed {len(failures)} failures")
                
                stats = analyzer.get_failure_statistics(failures)
                
                print("\n" + "=" * 80)
                print("FAILURE STATISTICS")
                print("=" * 80)
                print(f"\n📈 Total Failures: {stats['total_failures']}")
                print(f"🤖 Auto-fixable: {stats['auto_fixable_count']} ({stats['auto_fixable_percentage']:.1f}%)")
                
                print("\n📊 By Category:")
                for category, count in stats['by_category'].items():
                    print(f"   - {category}: {count}")
                
                print("\n⚠️  By Severity:")
                for severity, count in stats['by_severity'].items():
                    print(f"   - {severity}: {count}")
                
                if args.output:
                    output_data = {
                        "statistics": stats,
                        "failures": [
                            analyzer.generate_recovery_report(f)
                            for f in failures
                        ]
                    }
                    args.output.write_text(json.dumps(output_data, indent=2))
                    print(f"\n✅ Batch analysis saved to {args.output}")
            else:
                print("⚠️  No log files found")
        
        return 0
    
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
