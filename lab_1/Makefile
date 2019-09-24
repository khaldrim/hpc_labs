INC_DIRS = -Iinclude
CFLAGS = -c -Wall

all: clean obj/main.o obj/functions.o obj/secuential.o obj/simd.o dilation run

dilation:
	@ gcc obj/*.o -o bin/dilation
	@ echo "Compilated."

obj/main.o: src/main.c
	@ gcc  $(CFLAGS) $(INC_DIRS) src/main.c -o obj/main.o

obj/functions.o: src/functions.c
	@ gcc  $(CFLAGS) $(INC_DIRS) src/functions.c -o obj/functions.o

obj/secuential.o: src/secuential.c
	@ gcc  $(CFLAGS) $(INC_DIRS) src/secuential.c -o obj/secuential.o

obj/simd.o: src/simd.c
	@ gcc  $(CFLAGS) $(INC_DIRS) src/simd.c -o obj/simd.o -msse4.2

clean:
	@ echo "Cleaned."
	@ rm -rf obj/*
	@ rm -rf bin/*
	@ rm -rf test/*

run:
	@ echo "Running..."
	# @ ./bin/dilation -i ./data/circulos.raw -s ./test/circulos_secuential.raw -p ./test/circulos_simd.raw -N 256
	@ ./bin/dilation -i ./data/example14x14.raw -s ./test/example_secuential.raw -p ./test/example_simd.raw -N 14 -D 1
	# @ ./bin/dilation -i ./data/pikachu256x256.raw -s ./test/pikachu_secuential.raw -p ./test/pikachu_simd.raw -N 256 -D 1
