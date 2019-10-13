INC_DIRS = -Iinclude
CFLAGS = -c -Wall

all: clean obj/mandelbrot.o obj/mandelbrotp.o mandelbrot mandelbrotp run

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

run:
	@ echo "Running Mandelbrot"
	# @ time ./bin/mandelbrotp -i 500 -a 2 -b 1 -c 3 -d 4 -s 0.001 -f ./results/parallel.raw -t 6
	# @ time ./bin/mandelbrotp -i 500 -a -1 -b -1 -c 1 -d 1 -s 0.001 -f ./results/parallel.raw -t 8
	# @ time ./bin/mandelbrotp -i 500 -a -0.748766713922161 -b 0.123640844894862 -c -0.748766707771757 -d 0.123640851045266 -s 0.00000000001 -f ./results/parallel.raw -t 8
	
	# @ time ./bin/mandelbrot -i 500 -a -1 -b -1 -c 1 -d 1 -s 0.001 -f ./results/secuential.raw
	# @ time ./bin/mandelbrot -i 500 -a -0.748766713922161 -b -0.748766713922161 -c 0.123640844894862 -d 0.123640844894862 -s 0.000000000001 -f ./results/secuential.raw