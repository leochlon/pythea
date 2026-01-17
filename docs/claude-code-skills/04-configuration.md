# Skills Configuration

## Restricting Tool Access

Limit which tools Claude can use when the Skill is active:

```yaml
---
name: reading-files-safely
description: Read files without making changes
allowed-tools: Read, Grep, Glob
---
```

Or using YAML list format:

```yaml
---
name: reading-files-safely
description: Read files without making changes
allowed-tools:
  - Read
  - Grep
  - Glob
---
```

## Running Skills in Forked Context

Execute a Skill in an isolated sub-agent context with separate conversation history:

```yaml
---
name: code-analysis
description: Analyze code quality and generate detailed reports
context: fork
agent: general-purpose
---
```

Available agent types:
- `general-purpose` - General tasks
- `Explore` - Codebase exploration
- Custom agents defined in `.claude/agents/`

## Defining Hooks

Run scripts during Skill lifecycle events:

```yaml
---
name: secure-operations
description: Perform operations with security checks
hooks:
  PreToolUse:
    - matcher: "Bash"
      hooks:
        - type: command
          command: "./scripts/security-check.sh $TOOL_INPUT"
          once: true
---
```

### Hook Types

| Hook | When It Runs |
|------|--------------|
| `PreToolUse` | Before a tool is executed |
| `PostToolUse` | After a tool completes |
| `Stop` | When Skill execution ends |

### Hook Options

| Option | Description |
|--------|-------------|
| `matcher` | Tool name to match (supports patterns) |
| `type` | Hook type (`command`) |
| `command` | Shell command to execute |
| `once` | Run only once per session (true/false) |

## Controlling Skill Visibility

### Model-Only Skill (Hidden from Menu)

```yaml
---
name: internal-review-standards
description: Apply internal code review standards when reviewing pull requests
user-invocable: false
---
```

### Disable Model Invocation

```yaml
---
name: manual-only-skill
description: Only runs when explicitly called
disable-model-invocation: true
---
```

## Specifying Models

Force a specific model when Skill is active:

```yaml
---
name: complex-analysis
description: Complex analysis requiring advanced reasoning
model: claude-opus-4-5-20251101
---
```

## Subagent Access to Skills

Custom subagents don't automatically inherit Skills. Grant access via:

```yaml
# .claude/agents/code-reviewer.md
---
name: code-reviewer
description: Review code for quality and best practices
skills: pr-review, security-check
---
```

## Skills in Plugins

Distribute Skills through plugins:

```
my-plugin/
├── .claude-plugin/
│   └── plugin.json
└── skills/
    └── my-skill/
        └── SKILL.md
```

### Plugin Configuration

Enable Skills through plugin marketplace:

```bash
/plugin marketplace add anthropics/skills
/plugin install document-skills@anthropic-agent-skills
```

## MCP Tool References

Use fully qualified tool names to avoid "not found" errors:

```markdown
Use the BigQuery:bigquery_schema tool to retrieve table schemas.
Use the GitHub:create_issue tool to create issues.

Format: ServerName:tool_name
```

## Precedence Rules

When multiple Skills exist with the same name:

1. **Enterprise Skills** override all others
2. **Personal Skills** override Project Skills
3. **Project Skills** override Plugin Skills

## Environment Variables in Scripts

Scripts bundled with Skills can access:

| Variable | Description |
|----------|-------------|
| `$TOOL_INPUT` | Input provided to the current tool |
| `$CLAUDE_SESSION_ID` | Current session identifier |
| Standard shell variables | PATH, HOME, etc. |
