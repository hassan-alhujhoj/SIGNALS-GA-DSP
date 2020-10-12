PDFFILES = report.pdf

all: $(PDFFILES) 

.SUFFIXES:

%.pdf: %.tex
	pdflatex $*
	bibtex $*
	pdflatex $*
	pdflatex $*

clean:
	-rm *.pdf *.aux *.dvi *.log *~ *.bbl *.blg

