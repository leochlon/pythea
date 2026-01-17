# Creating Claude Code Skills

## Quick Start: Create Your First Skill

### Step 1: Create the Skill Directory

```bash
mkdir -p ~/.claude/skills/explaining-code
```

### Step 2: Create `SKILL.md` File

```yaml
---
name: explaining-code
description: Explains code with visual diagrams and analogies. Use when explaining how code works, teaching about a codebase, or when the user asks "how does this work?"
---

When explaining code, always include:

1. **Start with an analogy**: Compare the code to something from everyday life
2. **Draw a diagram**: Use ASCII art to show the flow, structure, or relationships
3. **Walk through the code**: Explain step-by-step what happens
4. **Highlight a gotcha**: What's a common mistake or misconception?

Keep explanations conversational. For complex concepts, use multiple analogies.
```

### Step 3: Load and Verify

Ask Claude:
```
What Skills are available?
```

### Step 4: Test the Skill

Ask Claude a question that matches the Skill's description:
```
How does this code work?
```

Claude should automatically use your Skill.

## Skill File Structure

### Minimal Skill (Single File)

```
my-skill/
└── SKILL.md
```

### Multi-File Skill (With Supporting Resources)

```
my-skill/
├── SKILL.md              # Required: overview and navigation
├── reference.md          # Optional: detailed API docs (loaded as needed)
├── examples.md           # Optional: usage examples (loaded as needed)
└── scripts/
    └── helper.py         # Optional: utility script (executed, not loaded)
```

## SKILL.md Structure

Every Skill requires a `SKILL.md` file with two parts:

```yaml
---
name: your-skill-name
description: Brief description of what this Skill does and when to use it
---

# Your Skill Name

## Section 1
Instructions and guidance for Claude...

## Section 2
More detailed information...
```

## Progressive Disclosure Pattern

Use separate files for detailed materials to keep the main SKILL.md concise:

**SKILL.md should link to supporting files:**

```markdown
## Additional resources

- For complete API details, see [reference.md](reference.md)
- For usage examples, see [examples.md](examples.md)
- To validate input, run: `python scripts/helper.py input.txt`
```

**Key guidelines:**
- Keep references one level deep from SKILL.md
- Structure long reference files with table of contents
- Bundle utility scripts for zero-context execution
- Scripts are executed without loading code into context

## Skill Locations

### Personal Skills (All Projects)

```bash
~/.claude/skills/{skill-name}/SKILL.md
```

### Project Skills (Repository-Specific)

```bash
.claude/skills/{skill-name}/SKILL.md
```

### Automatic Discovery from Nested Directories

Claude Code automatically discovers Skills from nested `.claude/skills/` directories, supporting monorepo setups:

- File in `packages/frontend/`: Claude also looks in `packages/frontend/.claude/skills/`

## Testing Your Skill

### Verify Loading

```bash
claude --debug
```

This shows loading errors if the Skill fails to load.

### Test Across Models

Skills effectiveness depends on the underlying model:

| Model | Guidance Needed |
|-------|-----------------|
| **Haiku** | Needs more detail |
| **Sonnet** | Balanced approach |
| **Opus** | Less explanation needed |

Test all three for best results.
