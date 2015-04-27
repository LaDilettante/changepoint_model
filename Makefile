paper.pdf: paper.tex
	pdflatex paper
	pdflatex paper
	pdflatex paper

paper.tex: paper.ipynb
	ipython nbconvert --to latex paper.ipynb

clean:
	latexmk -CA

