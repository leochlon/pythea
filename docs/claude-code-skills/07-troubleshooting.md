# Skills Troubleshooting

## Skill Not Triggering

**Problem:** Created a Skill but Claude never uses it

### Solutions

1. **Check description specificity**

   Vague descriptions like "Helps with documents" don't trigger reliably. Include specific capabilities and use cases.

   ```yaml
   # Bad
   description: Helps with documents

   # Good
   description: Extract text and tables from PDF files, fill forms, merge documents. Use when working with PDF files or when the user mentions PDFs, forms, or document extraction.
   ```

2. **Include keywords users would mention**

   Think about what words users actually say when they need this Skill.

3. **Test with explicit requests**

   Ask Claude to do something that clearly matches the description.

---

## Skill Doesn't Load

**Problem:** Skills don't appear in list

### Solutions

1. **Check file path** (case-sensitive)

   ```
   Personal: ~/.claude/skills/my-skill/SKILL.md
   Project:  .claude/skills/my-skill/SKILL.md
   ```

   Note: The file must be named exactly `SKILL.md` (uppercase).

2. **Check YAML syntax**

   Frontmatter must start with `---` on line 1 (no blank lines before):

   ```yaml
   ---
   name: my-skill
   description: Does something useful
   ---
   ```

   Use spaces, not tabs for indentation.

3. **Run debug mode**

   ```bash
   claude --debug
   ```

   This shows loading errors.

4. **Verify required metadata**

   Both `name` and `description` fields are required.

---

## Skill Has Errors

### Check Dependencies

If Skill uses external packages, verify they're installed:

```bash
pip list | grep pdfplumber
```

### Check Script Permissions

```bash
chmod +x scripts/*.py
chmod +x scripts/*.sh
```

### Check File Paths

Use forward slashes (`scripts/helper.py`), not backslashes.

### Verify Bash Syntax

Scripts must be executable and properly formatted:

```bash
bash -n scripts/my-script.sh  # Syntax check
```

---

## Multiple Skills Conflict

**Problem:** Claude uses wrong Skill or seems confused

### Solution

Make descriptions more distinct with specific trigger terms:

```yaml
# Before (too similar)
skill1: description: Helps with data analysis
skill2: description: Processes data files

# After (distinct)
skill1: description: Analyze sales data in Excel files and CRM exports. Use for sales reports and revenue analysis.
skill2: description: Process server log files and system metrics. Use for debugging and performance monitoring.
```

---

## Plugin Skills Not Appearing

**Problem:** Installed plugin but Skills don't show

### Solution

Clear plugin cache and reinstall:

```bash
rm -rf ~/.claude/plugins/cache
/plugin install plugin-name@marketplace-name
```

---

## Hooks Not Running

**Problem:** Defined hooks but they don't execute

### Check Hook Structure

```yaml
hooks:
  PreToolUse:
    - matcher: "Bash"  # Must match tool name
      hooks:
        - type: command
          command: "./scripts/check.sh $TOOL_INPUT"
```

### Verify Script Exists and Is Executable

```bash
ls -la scripts/check.sh
chmod +x scripts/check.sh
```

### Check Matcher Pattern

The `matcher` must exactly match the tool name or use a valid pattern.

---

## Forked Context Not Working

**Problem:** `context: fork` doesn't seem to isolate

### Verify Configuration

```yaml
---
name: my-skill
description: Does analysis
context: fork
agent: general-purpose  # Required when using context: fork
---
```

### Check Agent Type

Valid agent types:
- `general-purpose`
- `Explore`
- Custom agents defined in `.claude/agents/`

---

## MCP Tools Not Found

**Problem:** Skill references MCP tools but they're not available

### Use Fully Qualified Names

```markdown
# Wrong
Use the query tool to run SQL.

# Correct
Use the DatabaseServer:query tool to run SQL.
```

Format: `ServerName:tool_name`

### Verify MCP Server Is Connected

Check that the MCP server is running and connected.

---

## Skill Loads But Doesn't Work Well

### Problem: Too Verbose

Keep SKILL.md under 500 lines. Move details to separate files.

### Problem: Instructions Too Vague

Claude needs specific guidance, not general principles:

```markdown
# Vague (bad)
Write good code.

# Specific (good)
Use TypeScript strict mode. Functions should be under 50 lines.
Always handle errors with try/catch.
```

### Problem: Wrong Model

Some Skills work better with certain models:

```yaml
model: claude-opus-4-5-20251101  # For complex reasoning
```

---

## Common Error Messages

### "Skill not found"

- Check file path is correct
- Verify `SKILL.md` filename (uppercase)
- Run `claude --debug` to see loading errors

### "Invalid YAML"

- Check for tabs (use spaces)
- Verify `---` markers
- Quote strings with special characters

### "Missing required field"

- Both `name` and `description` are required
- Check for typos in field names

### "Tool not allowed"

- Check `allowed-tools` configuration
- Verify tool name is spelled correctly

---

## Debug Checklist

1. [ ] File path is correct (`~/.claude/skills/name/SKILL.md`)
2. [ ] File is named exactly `SKILL.md`
3. [ ] YAML frontmatter starts on line 1
4. [ ] Both `name` and `description` present
5. [ ] No tabs in YAML (spaces only)
6. [ ] Scripts are executable (`chmod +x`)
7. [ ] Forward slashes in paths
8. [ ] Dependencies installed
9. [ ] MCP tools use qualified names
10. [ ] Description is specific enough

---

## Getting Help

### Debug Mode

```bash
claude --debug
```

### List Available Skills

Ask Claude: "What Skills are available?"

### Check Skill Status

Ask Claude: "Is the [skill-name] Skill loaded?"

### Official Resources

- [Claude Code Docs](https://code.claude.com/docs)
- [Skills Documentation](https://code.claude.com/docs/en/skills.md)
- [Anthropic Skills Repository](https://github.com/anthropics/skills)
