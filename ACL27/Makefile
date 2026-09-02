.PHONY: all figures clean

all: paper.pdf

paper.pdf: paper.tex sections/main.tex sections/appendix.tex \
	bibliography/references.bib
	mkdir -p build
	pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build paper.tex
	bibtex build/paper
	pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build paper.tex
	pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build paper.tex
	cp build/paper.pdf paper.pdf

figures:
	python ../scripts/plot_acl_evidence_figures.py
	python ../scripts/plot_acl_story_figures.py --out-dir figures --figures signature surface hidden audit

clean:
	rm -f build/paper.aux build/paper.bbl build/paper.blg \
		build/paper.log build/paper.out build/paper.pdf
	rm -f build/previews/*.png
