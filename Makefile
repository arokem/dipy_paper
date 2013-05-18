.PHONY: 1 2 bib dvi pdf all

1:
	cd paper && latex dipy_paper.tex 

dvi: 1

bib:
	cd paper && bibtex dipy_paper

pdf: 1 2 1 
	cd paper && dvips dipy_paper.dvi && ps2pdf dipy_paper.ps

2: 1 bib

all: pdf mv

clean: 
	rm build/* 

mv: 
	mv paper/dipy_paper.* build/ && mv build/dipy_paper.tex paper/