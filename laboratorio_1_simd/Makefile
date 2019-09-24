INC_DIRS = -Iinclude
CFLAGS = -c -Wall

all: clean obj/main.o obj/functions.o obj/secuential.o obj/simd.o dilation

dilation:
	@ gcc obj/*.o -o bin/dilation
	@ echo "Compilation finished. dilation is in bin/ folder."

obj/main.o: src/main.c
	@ gcc  $(CFLAGS) $(INC_DIRS) src/main.c -o obj/main.o

obj/functions.o: src/functions.c
	@ gcc  $(CFLAGS) $(INC_DIRS) src/functions.c -o obj/functions.o

obj/secuential.o: src/secuential.c
	@ gcc  $(CFLAGS) $(INC_DIRS) src/secuential.c -o obj/secuential.o

obj/simd.o: src/simd.c
	@ gcc  $(CFLAGS) $(INC_DIRS) src/simd.c -o obj/simd.o -msse4.2

clean:
	@ echo "Clean bin/ results/ obj/ folders."
	@ rm -rf obj/
	@ rm -rf bin/
	@ rm -rf results/
	@ mkdir obj/ bin/ results/

# run:
# 	@ ./bin/dilation -i ./data/example14x14.raw -s ./results/example_secuential.raw -p ./results/example_simd.raw -N 14 -D 1