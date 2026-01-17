# Claude Code Skills Overview

## What Are Skills?

**Claude Code Skills** are modular capabilities that extend Claude's functionality by packaging instructions, metadata, and optional resources (scripts, templates) that Claude loads automatically when relevant to your request.

## Key Concept: Model-Invoked Capabilities

- Skills are **not** explicitly called like slash commands
- Claude **automatically decides** when to use a Skill based on your request
- Claude uses the Skill's `description` field to determine relevance
- When triggered, Claude loads the full `SKILL.md` content into context

## Three Levels of Loading (Progressive Disclosure)

| Level | When Loaded | Token Cost | Content |
|-------|-------------|------------|---------|
| **Level 1: Metadata** | Always (at startup) | ~100 tokens per Skill | `name` and `description` from YAML frontmatter |
| **Level 2: Instructions** | When Skill is triggered | Under 5k tokens | SKILL.md body with instructions and guidance |
| **Level 3+: Resources** | As needed | Effectively unlimited | Bundled files executed via bash without loading contents into context |

## The Skill Discovery Process

When you send a request, Claude follows these steps:

1. **Discovery**: Claude loads only the name and description of each available Skill (keeps startup fast)
2. **Activation**: When your request matches a Skill's description, Claude asks to use the Skill. You see a confirmation prompt
3. **Execution**: Claude follows the Skill's instructions, loading referenced files or running bundled scripts as needed

## Three Invocation Methods

### 1. Automatic Discovery (Model-Invoked)
Claude reads the Skill's description and automatically loads it when relevant.

```
User: "How does this code work?"
Claude: "I'll use the explaining-code Skill to help with this"
```

### 2. Manual Invocation (Slash Command)
You explicitly type `/skill-name` in the prompt.

```
/explaining-code show me how authentication works
```

### 3. Programmatic Invocation (Via Skill Tool)
Claude calls the Skill via the internal `Skill` tool when it determines it would improve the response.

## Where Skills Live

| Location | Path | Scope | Precedence |
|----------|------|-------|------------|
| **Enterprise** | Platform-specific | All org users | 1st (highest) |
| **Personal** | `~/.claude/skills/{skill-name}/SKILL.md` | You, across all projects | 2nd |
| **Project** | `.claude/skills/{skill-name}/SKILL.md` | Anyone in repository | 3rd |
| **Plugin** | `skills/{skill-name}/SKILL.md` inside plugin | Plugin users | 4th (lowest) |

## Skills vs. Other Customization Options

| Use This | When You Want | Triggered By |
|----------|---------------|--------------|
| **Skills** | Give Claude specialized knowledge | Claude chooses when relevant |
| **Slash Commands** | Create reusable prompts | You type `/command` |
| **CLAUDE.md** | Set project-wide instructions | Loaded into every conversation |
| **Subagents** | Delegate tasks to separate context | Claude delegates or you invoke |
| **Hooks** | Run scripts on events | Fires on specific tool events |
| **MCP servers** | Connect to external tools/data | Claude calls MCP tools as needed |

### Key Distinctions

**Skills vs. Subagents:**
- Skills add knowledge to current conversation
- Subagents run in separate context with own tools
- Use Skills for guidance; use subagents for isolation

**Skills vs. MCP:**
- Skills tell Claude *how* to use tools
- MCP *provides* the tools
- Example: MCP connects to database; Skill teaches data model

**Skills vs. Slash Commands:**
- Skills are automatic (Claude decides)
- Slash commands are manual (you type explicitly)
- Skills share knowledge; commands run fixed prompts
