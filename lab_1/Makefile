all: main.o functions.o
	gcc main.o	functions.o -o dilation -Wall
	rm *.o

main.o: src/main.o
	gcc -c src/main.c

functions.o: src/functions.o
	gcc -c src/functions.c
