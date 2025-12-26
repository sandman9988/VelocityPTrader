# Pull Request

## 📋 Description
<!-- Provide a brief description of your changes -->

## 🔗 Related Issues
<!-- Link to any related issues using #issue_number -->
Fixes #
Closes #
Related to #

## 🧪 Type of Change
<!-- Mark the relevant option with an 'x' -->

- [ ] 🐛 Bug fix (non-breaking change which fixes an issue)
- [ ] ✨ New feature (non-breaking change which adds functionality)
- [ ] 💥 Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] 📚 Documentation (changes to documentation only)
- [ ] 🧹 Code cleanup (refactoring, formatting, removing unused code, etc.)
- [ ] ⚡ Performance improvement
- [ ] 🔒 Security enhancement
- [ ] 🧪 Test improvement

## 🎯 Component Impact
<!-- Mark which components are affected -->

- [ ] 📊 Data Pipeline (MT5 Connection)
- [ ] 🤖 Agent System (BERSERKER/SNIPER)
- [ ] 💰 Trading Engine
- [ ] 📈 Performance Dashboard
- [ ] 📝 Logging System
- [ ] ⚙️ Configuration
- [ ] 🧪 Testing Framework
- [ ] 📖 Documentation

## ✅ Checklist

### Code Quality
- [ ] My code follows the project's style guidelines (black, isort, ruff)
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] My changes generate no new warnings or errors

### Testing
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] I have tested this change with real MT5 data (if applicable)
- [ ] I have verified this change doesn't break existing functionality

### Security & Performance
- [ ] My changes don't introduce security vulnerabilities
- [ ] I have considered performance implications
- [ ] No sensitive data (passwords, keys, etc.) is hardcoded
- [ ] MT5 connection security is maintained

### Documentation
- [ ] I have made corresponding changes to the documentation
- [ ] My changes are backward compatible OR I have documented breaking changes
- [ ] I have updated the changelog (if applicable)

## 🧪 Testing Details

### Test Environment
<!-- Describe your testing environment -->
- **OS**: 
- **Python Version**: 
- **MT5 Terminal**: 
- **Broker**: Vantage International Demo (required)

### Test Cases
<!-- Describe what you tested -->
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing performed
- [ ] Edge cases considered

### Performance Testing
<!-- If applicable, provide performance metrics -->
- **Before**: 
- **After**: 
- **Improvement**: 

## 📸 Screenshots/Logs
<!-- Add screenshots for UI changes or relevant log outputs -->

## 🚀 Deployment Considerations
<!-- Any special deployment requirements or migration steps -->

- [ ] Database migrations required
- [ ] Configuration updates needed
- [ ] Dependencies added/updated
- [ ] Environment variables changed

## 📝 Additional Notes
<!-- Any additional information that reviewers should know -->

## 🔍 Review Guidelines

### For Reviewers
Please verify:
1. **Security**: No credentials exposed, secure MT5 connection
2. **Performance**: No significant performance degradation
3. **Testing**: Adequate test coverage for changes
4. **Documentation**: Clear and accurate documentation
5. **Trading Logic**: Changes align with physics-based approach
6. **Code Quality**: Follows established patterns and standards

### Risk Assessment
<!-- Mark the risk level -->
- [ ] 🟢 Low risk (documentation, minor fixes, tests)
- [ ] 🟡 Medium risk (new features, refactoring)
- [ ] 🔴 High risk (core trading logic, security changes)

---

**By submitting this pull request, I confirm that:**
- [ ] I have the right to submit this contribution
- [ ] This contribution is my original work or properly attributed
- [ ] I understand this code will be used in a financial trading system
- [ ] I have tested thoroughly and considered potential financial impact