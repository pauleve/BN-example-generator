CLINGO_OPTS =
PSIZE = 1

all: gence4

gence%:
	rm -f solution.txt
	python main.py $(PSIZE) $* > $@.asp
	clingo $(CLINGO_OPTS) $@.asp | bash save-last-answer solution.txt | grep -v 'edge('
	test -s solution.txt
	make extract

extract:
	sed -i 's/ /\n/g' solution.txt
	(echo 'digraph{'; cat solution.txt | grep 'edge(r,' | sed -e 's/edge(r,//' -e 's/))/"/' -e 's/),(/" -> "/' -e 's/(/"/' -e 's/,//g' ; echo '}') > reduced.dot
	(echo 'digraph{'; cat solution.txt | grep 'edge((' |grep -v 'a'| sed -e 's/edge(//' -e 's/))/"/' -e 's/),(/" -> "/' -e 's/(/"/' -e 's/,//g' ; echo '}') > initial.dot
	-python extract_bn.py initial.dot
	-python extract_bn.py reduced.dot
	make dots

dots: $(patsubst %.dot,%.pdf, $(wildcard *.dot))
%.pdf: %.dot
	dot -Tpdf $< > $@
