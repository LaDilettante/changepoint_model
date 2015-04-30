paper.pdf: paper.tex
	pdflatex paper
	pdflatex paper
	pdflatex paper

paper.tex: paper.ipynb
	ipython nbconvert --to=latex --template=latex_nocode.tplx paper.ipynb

clean:
	latexmk -CA

