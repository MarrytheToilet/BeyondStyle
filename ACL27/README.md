# BeyondStyle ACL manuscript

The manuscript directory contains only active submission material:

- `paper.tex` — document preamble and assembly;
- `sections/` — eight-page main paper and two appendix chapters;
- `tables/` — the three active table fragments;
- `figures/` — publication-ready vector PDFs;
- `bibliography/references.bib` — active references;
- `acl.sty` and `acl_natbib.bst` — ACL formatting files;
- `build/` — disposable LaTeX output and PNG figure previews.

Rebuild from this directory:

```bash
make figures
make all
```

The final submission PDF is `paper.pdf`.  The main paper is limited to
eight pages before references; appendices follow the bibliography.
