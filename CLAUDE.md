# CLAUDE.md

## Repository overview

This repository contains several lecture/demo projects. Most subprojects include:

- Quarto source (`.qmd`)
- Rendered HTML lectures/slides (`.html` + supporting `*_files` assets)
- Jupyter notebooks (`.ipynb`)

### Main folders

- `Presentation/`: presentation deck (`presentation.qmd`, `presentation.html`) and assets.
- `ddmec/`: DDMEC lecture materials.
- `functional_kl/`: Functional KL lecture + code, checkpoints, and datasets.
- `infosedd/`: INFOSEDD lecture and notebook.
- `soi/`: SOI lecture/slides and notebooks.
- `tende/`: TENDE lecture and notebooks (including a Colab-oriented notebook).

## Notebook index

- `ddmec/notebook.ipynb`
- `infosedd/infosedd_lecture.ipynb`
- `soi/notebook.ipynb`
- `soi/notebook_soi_vbn.ipynb`
- `tende/notebook.ipynb`
- `tende/TENDE_notebook_colab.ipynb`

## Open notebooks in Google Colab

### Option A: GitHub-hosted notebook URL

Use:

`https://colab.research.google.com/github/<ORG>/<REPO>/blob/<BRANCH>/<PATH_TO_NOTEBOOK>.ipynb`

Example:

`https://colab.research.google.com/github/<ORG>/<REPO>/blob/main/tende/TENDE_notebook_colab.ipynb`

### Option B: Upload local notebook directly

1. Go to https://colab.research.google.com/
2. Open the **Upload** tab
3. Upload any `.ipynb` from this repository

## Open HTML files directly (with quick links)

Start a local server from repo root:

```bash
python3 -m http.server 8000
```

Then click one of these direct links:

- 🔘 [Open Presentation HTML](http://localhost:8000/Presentation/presentation.html)
- 🔘 [Open DDMEC lecture](http://localhost:8000/ddmec/lecture.html)
- 🔘 [Open Functional KL lecture](http://localhost:8000/functional_kl/lecture.html)
- 🔘 [Open INFOSEDD lecture](http://localhost:8000/infosedd/lecture.html)
- 🔘 [Open SOI lecture](http://localhost:8000/soi/lecture.html)
- 🔘 [Open TENDE lecture](http://localhost:8000/tende/lecture.html)

This method preserves relative asset loading (e.g., `lecture_files/`).

## Maintenance notes

- Prefer editing `.qmd` sources and then re-rendering HTML.
- Keep notebook + rendered HTML outputs aligned when updating lecture content.
