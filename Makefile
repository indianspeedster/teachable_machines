SOURCES := $(wildcard *.md)
NBS := $(SOURCES:.md=.ipynb)

%.ipynb: %.md
	pandoc --embed-resources --standalone --wrap=none  $< -o $@
	sed -i 's/attachment://g' edge_cpu_inference.ipynb 

all: $(NBS)

clean: 
	rm -f $(NBS)