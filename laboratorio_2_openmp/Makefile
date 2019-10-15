INC_DIRS = -Iinclude
CFLAGS = -c -Wall

all: clean obj/mandelbrot.o obj/mandelbrotp.o mandelbrot mandelbrotp

mandelbrot:
	@ gcc obj/mandelbrot.o -o bin/mandelbrot -lm
	@ echo "Compilation finished. mandelbrot is in bin/ folder."

mandelbrotp:
	@ gcc obj/mandelbrotp.o -o bin/mandelbrotp -lm -fopenmp
	@ echo "Compilation finished. mandelbrotp is in bin/ folder."

obj/mandelbrot.o: src/mandelbrot.c
	@ gcc  $(CFLAGS) $(INC_DIRS) src/mandelbrot.c -o obj/mandelbrot.o

obj/mandelbrotp.o: src/mandelbrotp.c
	@ gcc  $(CFLAGS) $(INC_DIRS) src/mandelbrotp.c -o obj/mandelbrotp.o -fopenmp

clean:
	@ echo "Clean bin/ results/ obj/ folders."
	@ rm -rf obj/
	@ rm -rf bin/
	@ rm -rf results/
	@ mkdir obj/ bin/ results/