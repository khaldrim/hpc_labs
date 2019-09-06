# make					builds executable file 'dilation'
# make clean		removes all .o and executable file

# define C compiler to use
CC = gcc

# define any compile-time flags
CFLAGS = -Wall -g

# define directory containing header files
INCLUDES = -I./include

# define source files
SRCS = ./src/main.c

# define .o files
OBJS = $(SRCS:.c=.o)

# define executable file
MAIN = dilation

all:	$(MAIN)
	@echo Laboratorio N°1 Compilado como dilation

$(MAIN): $(OBJS)
	$(CC) $(CFLAGS) $(INCLUDES) -o $(MAIN) $(OBJS)

.c.o:
	$(CC) $(CFLAGS) $(INCLUDES) -c $<	-o $@

clean:
	$(RM) *.o *~ $(MAIN)

depend: $(SRCS)
	makedepend $(INCLUDES) $^
