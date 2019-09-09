all: main.o functions.o secuential.o simd.o
	gcc obj/main.o obj/functions.o obj/secuential.o obj/simd.o -o dilation -Wall -msse3

main.o:
	gcc -c src/main.c -o obj/main.o

functions.o:
	gcc -c src/functions.c -o obj/functions.o

secuential.o:
	gcc -c src/secuential.c -o obj/secuential.o

simd.o:
	gcc -c src/simd.c -o obj/simd.o
