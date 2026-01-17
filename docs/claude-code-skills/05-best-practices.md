# Best Practices for Skill Development

## 1. Write Effective Descriptions

The description is **critical** for Skill discovery. Claude uses it to decide when to activate your Skill.

**Good descriptions answer two questions:**
1. What does this Skill do? (List specific capabilities)
2. When should Claude use it? (Include trigger terms)

### Good Example

```yaml
description: Extract text and tables from PDF files, fill forms, merge documents. Use when working with PDF files or when the user mentions PDFs, forms, or document extraction.
```

### Bad Examples

```yaml
description: Helps with documents
description: Processes data
description: Does stuff with files
```

### Important Rules

Always write descriptions in **third person**:
- Good: "Processes Excel files and generates reports"
- Avoid: "I can help you process Excel files"
- Avoid: "You can use this to process Excel files"

## 2. Keep SKILL.md Concise

**Target: Under 500 lines** for optimal performance

**Principle:** Claude is already very smart. Only add context Claude doesn't have.

### Concise Example (~50 tokens)

```markdown
## Extract PDF text

Use pdfplumber for text extraction:

```python
import pdfplumber
with pdfplumber.open("file.pdf") as pdf:
    text = pdf.pages[0].extract_text()
```
```

### Verbose Example (~150 tokens - avoid)

```markdown
## Extract PDF text

PDF (Portable Document Format) files are a common file format that contains
text, images, and other content. To extract text from a PDF, you'll need to
use a library. There are many libraries available for PDF processing, but we
recommend pdfplumber because it's easy to use and handles most cases well...
```

## 3. Use Progressive Disclosure

Use separate files for detailed materials:

```
my-skill/
├── SKILL.md (required - overview and navigation)
├── reference.md (loaded when needed)
├── examples.md (loaded when needed)
└── scripts/
    └── helper.py (executed, not loaded)
```

**Key guidelines:**
- Keep references one level deep from SKILL.md
- Structure long reference files with table of contents
- Bundle utility scripts for zero-context execution
- Scripts are executed without loading code into context

## 4. Use Consistent Naming

**Recommended: Gerund form** (verb + -ing):
```
processing-pdfs
analyzing-spreadsheets
managing-databases
writing-documentation
```

**Avoid:**
- Vague names: `helper`, `utils`, `tools`
- Generic: `documents`, `data`, `files`
- Reserved words: `anthropic-helper`, `claude-tools`

## 5. Set Appropriate Degrees of Freedom

Match specificity to task fragility:

| Freedom Level | When to Use | Example |
|---------------|-------------|---------|
| **High** (text instructions) | Multiple approaches valid | Code review processes |
| **Medium** (pseudocode/parameterized scripts) | Preferred pattern exists | Report generation templates |
| **Low** (specific scripts, no parameters) | Operations fragile, consistency critical | Database migrations |

## 6. Common Effective Patterns

### Template Pattern (For Strict Requirements)

```markdown
## Report structure

ALWAYS use this exact template:

```markdown
# [Title]

## Executive summary
[One-paragraph overview]

## Key findings
- Finding 1
- Finding 2
- Finding 3

## Recommendations
1. Specific actionable recommendation
2. Specific actionable recommendation
```
```

### Examples Pattern (For Input/Output Clarity)

```markdown
## Commit message format

**Example 1:**
Input: Added user authentication with JWT tokens
Output:
```
feat(auth): implement JWT-based authentication

Add login endpoint and token validation middleware
```
```

### Workflow Pattern (For Multi-Step Tasks)

```markdown
## Research synthesis workflow

- [ ] Step 1: Read all source documents
- [ ] Step 2: Identify key themes
- [ ] Step 3: Cross-reference claims
- [ ] Step 4: Create structured summary
- [ ] Step 5: Verify citations
```

## 7. Avoid Anti-Patterns

**Don't:**
- Use Windows-style paths (`scripts\helper.py`) - use forward slashes
- Include time-sensitive information
- Offer too many options (provide default with escape hatch)
- Use deeply nested file references
- Assume packages are pre-installed without listing them
- Use "voodoo constants" (unexplained magic values)

## 8. Test With Multiple Models

Skills work with different models. Test with:
- **Claude Haiku** (needs more detail)
- **Claude Sonnet** (balanced)
- **Claude Opus** (less explanation needed)

## 9. Evaluation-Driven Development

1. Run Claude on representative tasks without Skill
2. Document specific failures or missing context
3. Create three test scenarios
4. Write minimal instructions addressing gaps
5. Iterate based on real usage

## 10. Security Considerations

### Important Warning

Only use Skills from trusted sources. Malicious Skills can:
- Direct Claude to invoke harmful tool operations
- Access and exfiltrate sensitive data
- Perform unauthorized system access
- Execute dangerous code

### Best Practices

1. **Audit thoroughly**: Review all files in the Skill
2. **Avoid external sources**: Be wary of Skills that fetch from external URLs
3. **Check for unusual patterns**: Look for unexpected network calls or file access
4. **Treat like software installation**: Only use trusted, verified Skills

### Secure Skill Practices

**Don't hardcode sensitive information:**

```yaml
# WRONG
password: "secret123"
api_key: "sk-..."
```

**Use appropriate MCP connections instead:**

```markdown
Use the SecureVault:get_secret tool to retrieve credentials:

SecureVault:get_secret database_password
```

## Checklist Before Sharing a Skill

### Core Quality
- [ ] Description is specific with key terms
- [ ] Description explains what it does AND when to use it
- [ ] SKILL.md body is under 500 lines
- [ ] Additional details in separate files
- [ ] No time-sensitive information
- [ ] Consistent terminology
- [ ] Examples are concrete
- [ ] File references one level deep
- [ ] Progressive disclosure patterns used
- [ ] Workflows have clear steps

### Code and Scripts
- [ ] Scripts solve problems, not punt to Claude
- [ ] Error handling explicit
- [ ] No unexplained magic values
- [ ] Required packages listed and verified
- [ ] No Windows-style paths
- [ ] Validation/verification for critical operations

### Testing
- [ ] Tested with Haiku, Sonnet, Opus
- [ ] At least three real-world evaluations
- [ ] Team feedback incorporated
