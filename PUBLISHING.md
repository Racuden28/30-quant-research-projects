# How to publish (about 10 minutes, once)

Everything is ready. These are the only steps that need you, because they need your accounts.

## 1. GitHub repository
1. Sign in at github.com and create a new **public** repository named `30-quant-research-projects`
   (no README, no license; the repo already has them).
2. In this folder, run:
   ```bash
   git remote add origin https://github.com/<your-github-username>/30-quant-research-projects.git
   git push -u origin main
   ```
3. Replace `<your-github-username>` in `README.md`, `docs/build.py` and
   `01-monte-carlo-calibration/LINKEDIN_POST.md`, run `python docs/build.py`, commit and push.

## 2. Website (GitHub Pages)
Repository → Settings → Pages → Source: *Deploy from a branch* → Branch `main`, folder `/docs` → Save.
After a minute the site is live at `https://<your-github-username>.github.io/30-quant-research-projects/`.

## 3. LinkedIn
* **Featured** section: add the website link (it presents all 30 projects as they appear).
* **Projects** section: one entry per project, linking to its article page.
* **Post**: text and image are in `01-monte-carlo-calibration/LINKEDIN_POST.md`.

## 4. Optional: wider research audience
* **SSRN**: upload the article as a working paper (export the article page to PDF from your browser).
  Category: Econometrics / Risk Management.
* **Medium or Substack**: paste the article; the figures are in `01-monte-carlo-calibration/figures/`.

## Adding the next project
Create `NN-project-name/` with the same layout, fill its row in `PROJECTS` in `docs/build.py`
and in the table in `README.md`, run `python docs/build.py`, commit and push.
