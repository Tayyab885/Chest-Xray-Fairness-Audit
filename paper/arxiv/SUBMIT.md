# arXiv submission notes

Status: not submitted. The compiled PDF (`chest-xray-fairness.pdf`) is
self-archived in this repository and stands on its own. These notes are kept
only as a record of the source package, in case the preprint is submitted later.

Source for the preprint. arXiv rebuilds the PDF from source on upload, so submit
the source files, not the compiled PDF.

## Files to upload
- `chest-xray-fairness.tex`
- `gradcam_cardiomegaly_sex.png`

That is the whole package. The bibliography is inline (`thebibliography`), so no
`.bib` file is needed.

## Steps
1. Sign in at https://arxiv.org (create an account first if needed).
2. New submission. Primary category: `eess.IV` (Image and Video Processing).
   Cross-list `cs.LG` (Machine Learning).
3. First-time submitters to these categories may need an endorsement. If arXiv
   asks, request one from a co-author or an already-endorsed colleague.
4. Upload the two files above. Let arXiv compile. Confirm the built PDF matches
   `chest-xray-fairness.pdf` in this folder (title, three tables, one figure,
   three references).
5. Fill in title, author, and paste the abstract from the paper.
6. License: CC BY 4.0 is a reasonable default for a portfolio preprint.
7. Submit. arXiv assigns the identifier after moderation (usually next business day).

## Local build (optional)
`tectonic chest-xray-fairness.tex` from this directory rebuilds the PDF.

## If it is ever submitted and goes live
Add the arXiv link to the repository README and cite the preprint ID in future work.
