# Claude Code Skills Documentation

> Comprehensive reference for Claude Code Skills as of January 2026

## Documentation Structure

| Document | Description |
|----------|-------------|
| [Overview](./01-overview.md) | What are Skills and how they work |
| [Creating Skills](./02-creating-skills.md) | Step-by-step guide to create custom Skills |
| [SKILL.md Reference](./03-skillmd-reference.md) | Complete metadata and format specification |
| [Configuration](./04-configuration.md) | Settings, hooks, and advanced options |
| [Best Practices](./05-best-practices.md) | Guidelines for effective Skill development |
| [Examples](./06-examples.md) | Complete code examples |
| [Troubleshooting](./07-troubleshooting.md) | Common issues and solutions |

## Quick Start

```bash
# Create a Skill directory
mkdir -p ~/.claude/skills/my-skill

# Create SKILL.md
cat > ~/.claude/skills/my-skill/SKILL.md << 'EOF'
---
name: my-skill
description: Does X when user asks about Y
---

# My Skill Instructions

Your instructions here...
EOF
```

## Official Sources

- [Claude Code Skills Docs](https://code.claude.com/docs/en/skills.md)
- [Agent Skills Overview](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview)
- [Best Practices](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices)
- [Anthropic Skills Repository](https://github.com/anthropics/skills)
