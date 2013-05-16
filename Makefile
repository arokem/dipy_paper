.PHONY: 1 2 bib all

1 :
	cd paper && latex dipy_paper.tex

bib:
	bibtex dipy_paper

2 : 1 bib
	cd paper && latex dipy_paper.tex

all: 2

