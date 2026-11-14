<!-- DataSoft advanced research portfolio -->
<p align="center">
  <img src="assets/DataSoftOrg_Logo_512_512.jpg" alt="DataSoft Advanced AI Research Organization" width="300">
</p>

<h1 align="center">Advanced AI Research</h1>

<p align="center">
  Independent research, evaluation, and verification across advanced AI.
</p>

<p align="center">
  Intended website: <strong>datasoftcompany.org</strong> (deployment in progress) ·
  <a href="https://github.com/datasoftcompany-org">GitHub organization</a>
</p>

---

## About

This repository is the public catalogue for the organization's advanced AI research and engineering work. We study model behavior, transparency, evaluation, verification, and the reliability of intelligent systems.

Our purpose is to improve understanding through careful evaluation and open research. We do not build or train new foundation models. We emphasize explicit hypotheses, reproducible methods, recorded evidence, and clear limits on what each result supports.

## Completed research

### Transparency Lab — H000

**Status:** Completed; recorded result: **PASS (C0/C2)**

- Graph size: **7 nodes, 7 edges**
- Focus: transparency evaluation
- The result records the outcome of H000 under the stated C0/C2 checks. It should not be read as a general claim about model transparency beyond that evaluation.

## Released research artifacts

### Oracle — v0.1

**Status:** Versioned research artifact

- Model: **RoBERTa-large**
- Benchmark: **GLUE SST-2**
- Scope: an initial benchmark artifact for sentiment classification.
- No benchmark score or broader performance claim is asserted here.

### AGENTS policy battery — v1.1

**Status:** Versioned policy and evaluation artifact

- Defines the **HOTFIX**, **ENGINEER**, **REFACTOR**, and **TEACH** operating modes.
- Provides a structured basis for evaluating agent behavior against task-specific working policies.

## Research principles

- State the research question, method, and evaluation conditions.
- Preserve evidence and distinguish measured results from interpretation.
- Report limitations and make claims no broader than the supporting evidence.
- Keep research artifacts reviewable and reproducible where the underlying materials permit.
- Label work by its actual status, including ongoing work and known limitations.

## Open collaboration

Researchers, engineers, and organizations are welcome to review the public work, identify gaps, propose collaborations, and contribute reproducibility efforts through [GitHub Issues](https://github.com/datasoftcompany-org/Advanced-AI-Research/issues).

## Repository contents

- `index.html` — English-language organization homepage source; deployment is in progress.
- `projects/index.html` — public research register summarizing records already documented in this repository.
- `business/index.html` — English-language draft for the distinct Industry & AI Futures platform intended for `datasoftcompany.biz`; not deployed.
- `assets/` — organization branding.
- [`research/`](research/README.md) — layout and reproduction template for new public research projects.
- Project-specific code, data, and reports will be linked or added when they are ready for public release.


## GitHub workflow

Start from a clean working tree. Update the local default branch and create a branch for the work.

```sh
git switch main
git pull --ff-only origin main
git switch -c research/<project-name>
```

Replace `<project-name>` with the project name before running the command. Stage the intended files, review `git diff --cached`, and commit on the project branch. Publish the branch with `git push -u origin HEAD`, then open a pull request targeting `main`. After the pull request is merged, switch back to `main` and run `git pull --ff-only origin main` again.

---

*This portfolio records the scope and status stated for each artifact; inclusion does not imply independent replication or peer review.*
