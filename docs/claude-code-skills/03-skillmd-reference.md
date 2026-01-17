# SKILL.md Reference

## Required Fields

| Field | Max Length | Requirements | Purpose |
|-------|------------|--------------|---------|
| `name` | 64 chars | Lowercase letters, numbers, hyphens only. No XML tags, no reserved words (anthropic, claude) | Unique identifier for the Skill |
| `description` | 1024 chars | Non-empty, no XML tags | How Claude discovers when to use the Skill |

## Optional Fields

| Field | Type | Purpose | Example |
|-------|------|---------|---------|
| `allowed-tools` | String or List | Tools Claude can use without asking permission | `allowed-tools: Read, Grep, Glob` |
| `model` | String | Specify which model to use when Skill is active | `model: claude-opus-4-5-20251101` |
| `context` | String | Run Skill in isolated sub-agent context | `context: fork` |
| `agent` | String | Specify agent type when `context: fork` is set | `agent: Explore` |
| `hooks` | Object | Define lifecycle hooks (PreToolUse, PostToolUse, Stop) | See hooks section |
| `user-invocable` | Boolean | Controls visibility in slash command menu (default: true) | `user-invocable: false` |

## String Substitutions

Skills support dynamic values:

| Variable | Description | Usage |
|----------|-------------|-------|
| `$ARGUMENTS` | All arguments passed when invoking the Skill | Appended automatically if not present |
| `${CLAUDE_SESSION_ID}` | Current session ID | Useful for logging or correlating output |

**Example:**

```yaml
---
name: session-logger
description: Log activity for this session
---

Log the following to logs/${CLAUDE_SESSION_ID}.log:

$ARGUMENTS
```

## Complete Frontmatter Example

```yaml
---
name: pdf-processing
description: Extract text and tables from PDF files, fill forms, merge documents. Use when working with PDF files or when the user mentions PDFs, forms, or document extraction.
allowed-tools:
  - Read
  - Bash(python:*)
model: claude-opus-4-5-20251101
context: fork
agent: general-purpose
user-invocable: false
hooks:
  PreToolUse:
    - matcher: "Bash"
      hooks:
        - type: command
          command: "./scripts/security-check.sh $TOOL_INPUT"
          once: true
---

# PDF Processing

[Your Skill content here]
```

## Visibility Control

| Setting | Slash Menu | `Skill` Tool | Auto-Discovery |
|---------|-----------|--------------|----------------|
| `user-invocable: true` (default) | Visible | Allowed | Yes |
| `user-invocable: false` | Hidden | Allowed | Yes |
| `disable-model-invocation: true` | Visible | Blocked | Yes |

## allowed-tools Syntax

### String Format

```yaml
allowed-tools: Read, Grep, Glob
```

### List Format

```yaml
allowed-tools:
  - Read
  - Grep
  - Glob
```

### Pattern Matching

```yaml
allowed-tools:
  - Read
  - Bash(python:*)  # Only allow python commands
```

## Hooks Structure

```yaml
hooks:
  PreToolUse:
    - matcher: "ToolName"
      hooks:
        - type: command
          command: "./script.sh $TOOL_INPUT"
          once: true  # Run only once per session
  PostToolUse:
    - matcher: "ToolName"
      hooks:
        - type: command
          command: "./after-script.sh"
  Stop:
    - type: command
      command: "./cleanup.sh"
```

## YAML Formatting Rules

1. Frontmatter must start with `---` on line 1 (no blank lines before)
2. Use spaces, not tabs for indentation
3. Close frontmatter with `---` on its own line
4. Strings with special characters should be quoted

**Correct:**

```yaml
---
name: my-skill
description: "Processes files: PDFs, Word docs, and spreadsheets"
---
```

**Incorrect:**

```yaml

---
name: my-skill
description: Processes files: PDFs, Word docs, and spreadsheets
---
```

## Name Conventions

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
