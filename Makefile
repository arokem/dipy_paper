.PHONY: 1 2 all

1 :
	cd paper && latex dipy_paper.tex

2 : 1
	cd paper && latex dipy_paper.tex

all: 2

