# Workflow Auto-Recovery System

## Overview

The Workflow Auto-Recovery System is an AI-powered solution for automatically detecting, analyzing, and recovering from CI/CD workflow failures. It reduces manual intervention and accelerates the development process by intelligently diagnosing issues and applying automated fixes when safe to do so.

## Architecture

### Components

1. **Workflow Failure Analyzer** (`src/utils/workflow_failure_analyzer.py`)
   - Analyzes workflow failure logs
   - Categorizes failures into predefined types
   - Determines severity and auto-fixability
   - Generates recovery reports

2. **AI Recovery Agent** (`src/utils/ai_recovery_agent.py`)
   - Diagnoses specific failure causes
   - Generates automated recovery actions
   - Creates fix scripts and PR descriptions
   - Executes safe automated fixes

3. **Auto-Recovery Workflow** (`.github/workflows/auto-recovery.yml`)
   - Triggers automatically on workflow failures
   - Downloads and analyzes failure logs
   - Executes recovery actions for auto-fixable issues
   - Creates PRs for fixes or issues for manual intervention

4. **CLI Tool** (`workflow_recovery_cli.py`)
   - Manual analysis and recovery tool
   - Batch processing of failure logs
   - Local testing of recovery actions

## Failure Categories

The system recognizes the following failure categories:

| Category | Auto-Fixable | Description |
|----------|--------------|-------------|
| `dependency_issue` | ✅ Partial | Missing or incompatible Python packages |
| `linting_error` | ✅ Yes | Code style violations (Black, isort, Ruff, flake8) |
| `syntax_error` | ❌ No | Python syntax errors |
| `test_failure` | ❌ No | Unit or integration test failures |
| `type_error` | ❌ No | Type checking errors (mypy) |
| `security_scan` | ❌ No | Security vulnerabilities detected |
| `build_error` | ❌ No | Build process failures |
| `timeout` | ✅ Yes | Operation timeouts |
| `network_error` | ✅ Yes | Transient network connectivity issues |
| `permission_error` | ✅ Partial | File or workflow permission issues |
| `resource_exhaustion` | ⚠️ Partial | Memory or disk space issues |
| `configuration_error` | ❌ No | Invalid configuration files |

## Recovery Severity Levels

- **LOW**: Safe to auto-fix without review
- **MEDIUM**: Can be attempted with moderate confidence, review recommended
- **HIGH**: Requires careful review before applying
- **CRITICAL**: Manual intervention always required

## Usage

### Automatic Recovery

The system automatically triggers when any monitored workflow fails:

1. Workflow fails → Auto-Recovery workflow triggered
2. Logs downloaded and analyzed
3. Failure categorized and recovery actions generated
4. If auto-fixable (e.g., linting errors):
   - Fixes applied automatically
   - PR created with changes
   - CI runs on fix PR
5. If manual intervention needed:
   - Issue created with analysis
   - Recovery suggestions provided

### Manual Analysis with CLI

#### Analyze a Single Failure

```bash
python workflow_recovery_cli.py analyze \
  --log workflow.log \
  --workflow "CI/CD Pipeline" \
  --job "super-lint" \
  --step "Run Black" \
  --output report.json
```

#### Generate Recovery Actions

```bash
python workflow_recovery_cli.py recover \
  --log workflow.log \
  --workflow "CI/CD Pipeline" \
  --job "super-lint" \
  --step "Run Black" \
  --output recovery_plan.json
```

#### Execute Recovery (Dry Run)

```bash
python workflow_recovery_cli.py execute \
  --log workflow.log \
  --workflow "CI/CD Pipeline" \
  --job "super-lint" \
  --step "Run Black" \
  --dry-run
```

#### Batch Analysis

```bash
python workflow_recovery_cli.py batch \
  --logs-dir ./workflow-logs \
  --workflow "CI/CD Pipeline" \
  --output batch_report.json
```

## Automated Fix Examples

### Linting Errors

**Detection:**
```
Error: Files would be reformatted by Black
Error: Import order violations detected by isort
```

**Automatic Fix:**
```bash
black src/ tests/
isort src/ tests/
ruff check --fix src/ tests/
```

**Result:** PR created with formatted code

### Dependency Issues

**Detection:**
```
ModuleNotFoundError: No module named 'pandas'
```

**Automatic Fix:**
```bash
echo 'pandas>=2.0.0' >> requirements.txt
```

**Result:** PR created with updated requirements.txt (requires review)

### Network Errors

**Detection:**
```
ConnectionError: Failed to download package from PyPI
```

**Automatic Fix:**
- Workflow automatically retried
- Caching enabled for future runs

**Result:** Workflow re-triggered

## Configuration

### Monitored Workflows

Edit `.github/workflows/auto-recovery.yml` to monitor additional workflows:

```yaml
on:
  workflow_run:
    workflows: ["Your Workflow Name", "Another Workflow"]
    types:
      - completed
```

### Custom Recovery Patterns

Add custom failure patterns in `src/utils/workflow_failure_analyzer.py`:

```python
FailurePattern(
    category=FailureCategory.CUSTOM_CATEGORY,
    severity=RecoverySeverity.MEDIUM,
    patterns=[
        r"Your error pattern here",
    ],
    description="Description of the failure",
    auto_fixable=True,
    fix_template="How to fix it"
)
```

### Custom Recovery Actions

Extend `src/utils/ai_recovery_agent.py` to add custom recovery logic:

```python
def _recover_custom_failure(self, failure: WorkflowFailure) -> List[RecoveryAction]:
    """Generate recovery actions for custom failure type"""
    actions = []
    
    # Your custom recovery logic here
    
    return actions
```

## Safety Features

1. **Severity-Based Gating**: High and critical severity issues never auto-recover
2. **Review Requirements**: PRs marked for review when confidence is low
3. **Dry-Run Support**: Test recovery actions without making changes
4. **Audit Trail**: All analysis and actions logged as artifacts
5. **Manual Override**: Issues created for human review when needed

## Metrics and Monitoring

The system tracks:

- Total failures analyzed
- Failures by category and severity
- Auto-recovery success rate
- Recovery action confidence levels
- Manual intervention requests

Access metrics via:
- Workflow run summaries
- Uploaded artifacts (JSON reports)
- Created issues and PRs

## Best Practices

1. **Review Auto-Created PRs**: Always review before merging, even for "safe" fixes
2. **Monitor Recovery Success**: Track which categories recover successfully
3. **Tune Patterns**: Add patterns for repo-specific failures
4. **Update Templates**: Improve recovery actions based on experience
5. **Test Locally**: Use CLI tool to test recovery before relying on automation

## Troubleshooting

### No Logs Downloaded

**Issue**: Auto-recovery can't download workflow logs

**Solution**: Ensure workflow has proper permissions:
```yaml
permissions:
  actions: read
  contents: write
```

### Recovery Actions Not Executing

**Issue**: Fixes not being applied automatically

**Check**:
1. Is the failure category marked as `auto_fixable`?
2. Is the severity LOW or MEDIUM?
3. Are there any permission issues in the repository?

### False Positive Auto-Fixes

**Issue**: System attempts to fix non-issues

**Solution**: Refine failure patterns to be more specific:
```python
patterns=[
    r"Very specific error pattern",  # More specific
    # r"error",  # Too broad
]
```

## Extending the System

### Adding New Failure Categories

1. Add category to `FailureCategory` enum
2. Create failure pattern in `_initialize_patterns()`
3. Implement recovery method in `AIRecoveryAgent`
4. Add workflow step for automated fixes (if applicable)

### Integrating with External Systems

The system can be extended to:
- Send Slack notifications on failures
- Create Jira tickets for manual interventions
- Post metrics to monitoring dashboards
- Trigger custom webhooks

Example integration point in auto-recovery workflow:

```yaml
- name: Send notification
  if: steps.analyze.outputs.failure_detected == 'true'
  run: |
    curl -X POST https://hooks.slack.com/... \
      -d '{"text": "Workflow failure detected: ${{ steps.analyze.outputs.category }}"}'
```

## Security Considerations

1. **Code Review**: All auto-fixes should undergo code review
2. **Secrets**: Never log or expose secrets in failure analysis
3. **Permissions**: Minimize workflow permissions to only what's needed
4. **Validation**: Validate all recovery actions before execution
5. **Audit**: Maintain audit trail of all automated changes

## Future Enhancements

- [ ] Machine learning for pattern detection
- [ ] Historical analysis for recurring failures
- [ ] Predictive failure prevention
- [ ] Integration with more CI/CD platforms
- [ ] Advanced natural language failure descriptions
- [ ] Automatic test generation for fixed issues

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the source code documentation
3. Create an issue in the repository
4. Review generated failure reports and recovery plans

## License

This auto-recovery system is part of the VelocityPTrader project and follows the same license.
