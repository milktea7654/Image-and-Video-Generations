# Image and Video Generations

NYCU Image and Video Generation coursework repository.

**Note**
- **Instructor**: 劉育綸
- **Semester**: 2025 Fall

## Lab

| Lab | Topic |
| --- | --- |
| Lab 1 | DDPM |
| Lab 2 | DDIM and LoRA |
| Lab 3 | Score Distillation Sampling |
| Lab 4 | Flow Matching |

## Repository Structure

```text
.
├── Lab1-DDPM/
├── Lab2-DDIM/
├── Lab3-Distillation/
├── Lab4-FlowMatching/
├── final-project/
├── .gitignore
└── .gitmodules
```

## Repository Layout

Labs are kept as normal folders in this repository. The final project uses external model repositories under `final-project/models/` as submodules.

```bash
git submodule update --init --recursive
```

## Folder Convention

- `README.md`: task overview, setup, commands, and submission notes.
- `requirements.txt`: Python dependencies.
- `assets/`: README images and visual results.
- `docs/`: assignment PDFs, result summaries, and written notes.
- `data/`: local datasets.
- `output/`, `outputs/`, `results/`: generated files, logs, and checkpoints.
