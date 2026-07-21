
CC = gcc
FLAGS = -fPIC -Wall -O2 -shared

SOURCES = $(wildcard *.c)
TARGETS = $(patsubst %.c,lib%.so,$(SOURCES))

all: $(TARGETS)

lib%.so: %.c
	$(CC) $(FLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)

.PHONY: all clean



#