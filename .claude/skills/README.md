# Agent skills

Skills are checked-in playbooks for long, error-prone maintenance jobs that we
would otherwise re-explain to a coding agent every time. Each subdirectory here
is one skill: a `SKILL.md` that tells the agent how to run the job, plus the
scripts and config the job needs.

Claude Code picks skills up automatically from `.claude/skills/` and exposes
each one as a slash command named after its directory. Nothing else in
`.claude/` is shared — `.gitignore` keeps `.claude/*` out of the repo and
re-includes only `.claude/skills/`, so local agent settings stay local.

These are maintainer tools, not part of the runtime. Nothing in this directory
is imported by `sglang_omni/`, and no CI job runs them.

## Available skills

| Skill | What it does | Who can run it |
|---|---|---|
| [`model-profiling`](model-profiling/SKILL.md) | Plans a layered profiling run for one model, waits for confirmation, then hands the GPU work to a background agent and checks its report. | Any sglang-omni dev container with free GPUs and the `omni` venv. |
| [`omni-gpu-deep-dive`](omni-gpu-deep-dive/SKILL.md) | Attributes GPU time in one pipeline stage to lines of `sglang_omni/` source from a mapping/formal trace pair. | Any sglang-omni dev container with a free GPU. |

`code-review/` holds the repository coding style guide that `CLAUDE.md` points
to; it is not a slash command.

CI threshold calibration is maintained separately in the private
[`sglang-omni-calibration` repository](https://github.com/zhaochenyang20/sglang-omni-calibration/tree/main/skills/calibrate-h100-ci).
Maintainers with access can find `calibrate-h100-ci`, its tests, and calibration
instructions there.

## Running one

Type the slash command in Claude Code from the repo root:

```
/model-profiling voicing_tts
```

Read the skill's `SKILL.md` first and keep a supervision terminal open
alongside the job.

## Adding a skill

Layout:

```
.claude/skills/<skill-name>/
├── SKILL.md          # required: frontmatter + the playbook
├── <tool>.py         # the actual work, runnable without an agent
└── models/ hosts/    # config, one file per model or host
```

`SKILL.md` frontmatter needs `name` (matching the directory) and
`description`. The description is the only thing an agent sees when deciding
whether to reach for the skill, so lead with the trigger, then the mechanism:

```yaml
---
name: model-profiling
description: Plan a layered profiling run for one model, stop for confirmation, then delegate the GPU work to a background agent.
---
```
