# Workflow Auto-Recovery Quick Start

## What is Workflow Auto-Recovery?

The Workflow Auto-Recovery system automatically detects, analyzes, and fixes CI/CD workflow failures in your GitHub Actions pipelines. It reduces manual intervention and speeds up development by intelligently handling common failure types.

## Quick Start

### 1. The System is Already Active! 🎉

The auto-recovery workflow is already monitoring these workflows:
- Comprehensive CI/CD Pipeline
- PostgreSQL Enterprise Testing
- CodeQL Security Analysis

When any of these workflows fail, the auto-recovery system will:
1. **Analyze** the failure automatically
2. **Categorize** the problem (linting, dependencies, network, etc.)
3. **Fix** it automatically (if safe to do so)
4. **Create a PR** with the fix, OR
5. **Create an issue** if manual intervention is needed

### 2. What Gets Auto-Fixed?

✅ **Automatically Fixed (No Review Needed)**
- Code formatting issues (Black, isort, Ruff)
- Import sorting
- Network errors (retries)
- Transient timeouts

⚠️ **Fixed with Review Required**
- Dependency version conflicts
- Permission issues
- Timeout configuration

❌ **Manual Intervention Required**
- Test failures
- Syntax errors
- Security vulnerabilities
- Build errors

### 3. Manual Analysis Tool

You can also analyze failures locally:

```bash
# Analyze a workflow log file
python workflow_recovery_cli.py analyze \
  --log path/to/workflow.log \
  --workflow "CI/CD Pipeline" \
  --job "super-lint" \
  --step "Run Black"

# Get recovery suggestions
python workflow_recovery_cli.py recover \
  --log path/to/workflow.log \
  --workflow "CI/CD Pipeline" \
  --job "super-lint" \
  --step "Run Black"

# Execute recovery (dry-run first!)
python workflow_recovery_cli.py execute \
  --log path/to/workflow.log \
  --workflow "CI/CD Pipeline" \
  --job "super-lint" \
  --step "Run Black" \
  --dry-run
```

### 4. Monitoring Auto-Recovery

#### Check Workflow Runs
1. Go to Actions tab in GitHub
2. Look for "Workflow Auto-Recovery" runs
3. Check the summary for analysis results

#### Check Created PRs
Auto-recovery PRs are labeled with:
- `auto-recovery`
- `workflow-fix`

Look for PRs with titles starting with "🤖 Auto-recovery:"

#### Check Created Issues
If manual intervention is needed, issues are created with:
- `workflow-failure`
- `needs-investigation`

Look for issues with titles starting with "🚨 Workflow Failure Requires Manual Intervention:"

### 5. Example Scenarios

#### Scenario 1: Linting Failure

**What Happens:**
1. Your PR fails CI because code isn't formatted
2. Auto-recovery workflow triggers
3. Runs `black`, `isort`, `ruff --fix` automatically
4. Creates PR with formatted code
5. You review and merge the PR

**Result:** Fixed in minutes without manual intervention!

#### Scenario 2: Test Failure

**What Happens:**
1. Your PR fails because a test is failing
2. Auto-recovery workflow triggers
3. Analyzes the failure
4. Creates an issue with:
   - Error details
   - Suggested recovery actions
   - Links to failed test logs

**Result:** You get a detailed issue to investigate and fix manually.

#### Scenario 3: Network Glitch

**What Happens:**
1. Workflow fails due to PyPI connection timeout
2. Auto-recovery triggers
3. Automatically retries the workflow

**Result:** Fixed automatically without any action needed!

### 6. Customizing Auto-Recovery

#### Add Custom Failure Patterns

Edit `src/utils/workflow_failure_analyzer.py`:

```python
FailurePattern(
    category=FailureCategory.CUSTOM,
    severity=RecoverySeverity.LOW,
    patterns=[
        r"Your custom error pattern",
    ],
    description="Description of your failure",
    auto_fixable=True,
    fix_template="How to fix it"
)
```

#### Monitor Additional Workflows

Edit `.github/workflows/auto-recovery.yml`:

```yaml
on:
  workflow_run:
    workflows: ["Your Custom Workflow", "Another Workflow"]
    types:
      - completed
```

### 7. Best Practices

1. **Always review auto-created PRs** before merging
2. **Monitor the auto-recovery workflow** for any issues
3. **Update failure patterns** as you encounter new issues
4. **Use the CLI tool** to test recovery locally before pushing
5. **Keep documentation updated** with new patterns

### 8. Troubleshooting

#### Auto-Recovery Workflow Doesn't Trigger

**Check:**
- Is the failed workflow in the monitored list?
- Does the auto-recovery workflow have proper permissions?

**Fix:** Add your workflow to the monitored list in auto-recovery.yml

#### No Logs Downloaded

**Check:** Workflow permissions

**Fix:** Add these permissions to your workflows:
```yaml
permissions:
  actions: read
  contents: read
```

#### Recovery PR Not Created

**Check:**
- Was the failure auto-fixable?
- Are there actual code changes to commit?

**Review:** Check the auto-recovery workflow logs for details

### 9. Getting Help

1. **View detailed documentation:** See `docs/WORKFLOW_AUTO_RECOVERY.md`
2. **Check workflow logs:** Review auto-recovery workflow runs
3. **Analyze locally:** Use the CLI tool to investigate
4. **Create an issue:** If something's not working as expected

### 10. Next Steps

- [ ] Review the first auto-recovery PR
- [ ] Monitor auto-recovery workflow runs
- [ ] Add custom failure patterns for your repo
- [ ] Share feedback on what works well
- [ ] Extend with additional recovery actions

## Questions?

See the full documentation at `docs/WORKFLOW_AUTO_RECOVERY.md` for:
- Architecture details
- All failure categories
- Advanced configuration
- Extension guide
- Security considerations

---

**Happy Auto-Recovering! 🤖✨**
