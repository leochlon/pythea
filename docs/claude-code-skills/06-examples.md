# Claude Code Skills Examples

## Example 1: Simple Skill (Single File)

**Goal:** Generate commit messages from git diffs

### Structure

```
commit-helper/
└── SKILL.md
```

### SKILL.md

```yaml
---
name: generating-commit-messages
description: Generates clear commit messages from git diffs. Use when writing commit messages or reviewing staged changes.
---

# Generating Commit Messages

## Instructions

1. Run `git diff --staged` to see changes
2. Generate a commit message with:
   - Summary under 50 characters
   - Detailed description explaining what and why
   - List of affected components

## Format

Use present tense and explain what and why, not how.

### Example

Input changes: Added JWT authentication to login endpoint
Output:
```
feat(auth): implement JWT-based authentication

Add login endpoint with JWT token validation middleware
for secure user authentication
```

## Best practices

- Use present tense
- Be specific about scope (e.g., feat(auth) not feat)
- Explain what and why, not how
```

---

## Example 2: Multi-File Skill with Progressive Disclosure

**Goal:** PDF processing with form filling

### Structure

```
pdf-processing/
├── SKILL.md
├── FORMS.md
├── REFERENCE.md
└── scripts/
    ├── analyze_form.py
    └── fill_form.py
```

### SKILL.md

```yaml
---
name: pdf-processing
description: Extract text and tables from PDF files, fill forms, merge documents. Use when working with PDF files, forms, or document extraction.
allowed-tools: Read, Bash(python:*)
---

# PDF Processing

## Quick start

Extract text with pdfplumber:

```python
import pdfplumber
with pdfplumber.open("doc.pdf") as pdf:
    text = pdf.pages[0].extract_text()
```

## Form filling

For detailed instructions, see [FORMS.md](FORMS.md).

**Quick example:**

```bash
python scripts/analyze_form.py input.pdf
python scripts/fill_form.py input.pdf fields.json output.pdf
```

## API reference

See [REFERENCE.md](REFERENCE.md) for complete pdfplumber and pdf2image methods.

## Requirements

```bash
pip install pypdf pdfplumber
```
```

### FORMS.md

```markdown
# Form Filling Guide

## Workflow

1. Analyze the form: `python scripts/analyze_form.py input.pdf`
2. Create field mapping in JSON
3. Fill the form: `python scripts/fill_form.py input.pdf fields.json output.pdf`

## Field mapping format

```json
{
  "field_name": "value",
  "customer_name": "John Doe",
  "signature": "signed"
}
```

## Common issues

- Field names are case-sensitive
- Date fields expect ISO format (YYYY-MM-DD)
- Checkbox fields use "Yes"/"No" or "On"/"Off"
```

### scripts/analyze_form.py

```python
#!/usr/bin/env python
import pdfplumber
import json
import sys

pdf_path = sys.argv[1]
with pdfplumber.open(pdf_path) as pdf:
    fields = {}
    for page_num, page in enumerate(pdf.pages):
        # Extract form fields
        if hasattr(page, 'annots') and page.annots:
            for annot in page.annots:
                if annot.get('field_name'):
                    fields[annot['field_name']] = {
                        'page': page_num,
                        'type': annot.get('field_type', 'unknown')
                    }
    print(json.dumps(fields, indent=2))
```

---

## Example 3: Skill with Hooks and Restricted Tools

**Goal:** Read-only file analysis with audit logging

### SKILL.md

```yaml
---
name: safe-code-analysis
description: Analyze code for patterns without modifying files. Use for code reviews, pattern analysis, or security scanning.
allowed-tools: Read, Grep, Glob
user-invocable: false
hooks:
  PostToolUse:
    - matcher: "Read"
      hooks:
        - type: command
          command: "echo 'Analyzed file: $TOOL_INPUT' >> /tmp/audit.log"
---

# Safe Code Analysis

Use this Skill for read-only code analysis. No file modifications permitted.

## Workflow

1. Read target files
2. Search for patterns
3. Generate report

## Security patterns to check

- Hardcoded credentials
- SQL injection vulnerabilities
- XSS vulnerabilities
- Insecure deserialization
```

---

## Example 4: Skill with Forked Context

**Goal:** Intensive code analysis in isolated context

### SKILL.md

```yaml
---
name: intensive-code-analysis
description: Perform in-depth code analysis in isolated context
context: fork
agent: Explore
---

# Intensive Code Analysis

This Skill runs in a forked sub-agent context for complex analysis tasks.

## Process

1. Scan entire codebase structure
2. Identify architectural patterns
3. Map dependencies
4. Generate comprehensive report

The analysis runs in isolation, keeping the main conversation clean.

## Output

Results are returned to the main conversation after analysis completes.
```

---

## Example 5: Database Query Skill with MCP

**Goal:** Query database using MCP tools

### SKILL.md

```yaml
---
name: database-querying
description: Query company database for reports and analysis. Use when user needs data from the database or asks about records, reports, or analytics.
---

# Database Querying

## Available tools

Use the fully qualified MCP tool names:

- `DatabaseServer:query` - Execute SQL queries
- `DatabaseServer:schema` - Get table schemas
- `DatabaseServer:tables` - List available tables

## Usage pattern

1. First, check available tables:
   ```
   DatabaseServer:tables
   ```

2. Get schema for relevant table:
   ```
   DatabaseServer:schema users
   ```

3. Execute query:
   ```
   DatabaseServer:query SELECT * FROM users WHERE active = true LIMIT 10
   ```

## Best practices

- Always use LIMIT to prevent large result sets
- Use parameterized queries when possible
- Check schema before writing complex queries
```

---

## Example 6: Code Review Skill

**Goal:** Standardized code review process

### SKILL.md

```yaml
---
name: code-reviewing
description: Review code for quality, security, and best practices. Use when reviewing pull requests, code changes, or performing code audits.
---

# Code Review Process

## Review checklist

### 1. Security
- [ ] No hardcoded secrets
- [ ] Input validation present
- [ ] SQL injection protection
- [ ] XSS prevention

### 2. Code quality
- [ ] Functions under 50 lines
- [ ] Clear variable names
- [ ] No magic numbers
- [ ] Error handling complete

### 3. Testing
- [ ] Unit tests for new code
- [ ] Edge cases covered
- [ ] Integration tests if needed

### 4. Documentation
- [ ] Public APIs documented
- [ ] Complex logic commented
- [ ] README updated if needed

## Output format

```markdown
## Code Review: [File/PR Name]

### Summary
[One paragraph overview]

### Issues Found
1. **[Severity]** - [Description]
   - Location: `file.py:123`
   - Suggestion: [How to fix]

### Positive Observations
- [What was done well]

### Recommendation
[Approve / Request Changes / Needs Discussion]
```
```

---

## Example 7: Session Logging Skill

**Goal:** Log session activity with dynamic session ID

### SKILL.md

```yaml
---
name: session-logging
description: Log all significant actions during this session for audit purposes.
---

# Session Logging

## Session ID

Current session: `${CLAUDE_SESSION_ID}`

## Log format

All actions should be logged to: `logs/${CLAUDE_SESSION_ID}.log`

## What to log

- File reads and writes
- Commands executed
- Decisions made
- Errors encountered

## Log entry format

```
[TIMESTAMP] [ACTION_TYPE] [DETAILS]
```

Example:
```
[2026-01-17T10:30:00Z] [FILE_READ] Read config.json
[2026-01-17T10:30:05Z] [COMMAND] npm install
[2026-01-17T10:30:10Z] [DECISION] Using TypeScript over JavaScript
```
```

---

## Built-in Skills Reference

Anthropic provides these Skills for common document tasks:

| Skill | ID | Capabilities |
|-------|-----|--------------|
| **PowerPoint** | `pptx` | Create presentations, edit slides |
| **Excel** | `xlsx` | Create spreadsheets, analyze data |
| **Word** | `docx` | Create documents, edit content |
| **PDF** | `pdf` | Generate formatted PDF documents |

### Invocation

```
/pptx Create a presentation about Q4 results
/xlsx Analyze the sales data in report.csv
/docx Write a project proposal
/pdf Generate a formatted report
```
