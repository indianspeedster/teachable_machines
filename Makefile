SOURCES := $(wildcard *.md)
NBS := $(SOURCES:.md=.ipynb)

%.ipynb: %.md
	pandoc --embed-resources --standalone --wrap=none  teachable_machines_on_edge.md -o teachable_machines_on_edge.ipynb
	sed -i 's/attachment://g' teachable_machines_on_edge.ipynb 
	

all: $(NBS)

clean: 
	rm -f $(NBS)
