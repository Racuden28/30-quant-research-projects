# How to publish (about 10 minutes, once)

Everything is ready. These are the only steps that need you, because they need your accounts.

## 1. GitHub repository and website (done)
The repository is https://github.com/Racuden28/30-quant-research-projects and GitHub Pages serves `docs/`
from `main` at https://racuden28.github.io/30-quant-research-projects/.

## 2. LinkedIn
* **Featured** section: add the website link (it presents all 30 projects as they appear).
* **Projects** section: one entry per project, linking to its article page.
* **Post**: text and image are in `01-monte-carlo-calibration/LINKEDIN_POST.md`.

## 3. Optional: wider research audience
* **SSRN**: upload the article as a working paper (export the article page to PDF from your browser).
  Category: Econometrics / Risk Management.
* **Medium or Substack**: paste the article; the figures are in `01-monte-carlo-calibration/figures/`.

## Adding the next project
Create `NN-project-name/` with the same layout, fill its row in `PROJECTS` in `docs/build.py`
and in the table in `README.md`, run `python docs/build.py`, commit and push.
