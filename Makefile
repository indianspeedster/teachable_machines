SOURCES := $(wildcard *.md)
NBS := $(SOURCES:.md=.ipynb)

%.ipynb: %.md
	pandoc --embed-resources --standalone --wrap=none  edge_cpu_inference.md -o edge_cpu_inference.ipynb
	sed -i 's/attachment://g' edge_cpu_inference.ipynb 
	

all: $(NBS)

clean: 
	rm -f $(NBS)