CC = g++
LD = g++
LDFLAGS = --no-warn
CFLAGS= --no-warn -O2
LIBS = -O2 -lfreenect -lGL -lGLU -lglut -lpthread
OBJECTS = danznect.o
PROG = danznect

all:$(PROG)

$(PROG): $(OBJECTS)
	$(LD) $(LDFLAGS) -o $(PROG) $(OBJECTS) $(LIBS)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< $(LIBS)

clean:
	rm -rf *.o $(PROG)

