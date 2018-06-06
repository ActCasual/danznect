CXX = g++
FLAGS= -std=c++11 -I /usr/include/libusb-1.0/
DEBUGFLAGS   = -O0 -g --no-warn
RELEASEFLAGS = -O2 --no-warn
LIBS = -lfreenect -lGL -lGLU -lglut -lpthread -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_photo
OBJECTS = danznect.o
PROG = danznect

%.o: %.cpp	
	$(CXX) $(FLAGS) -c $< $(LIBS)

.PHONY: release
release: FLAGS+=$(RELEASEFLAGS)
release: $(OBJECTS)
	$(CXX) $(FLAGS) -o $(PROG) $(OBJECTS) $(LIBS)

.PHONY: debug
debug: FLAGS+=$(DEBUGFLAGS)
debug: $(OBJECTS)
	$(CXX) $(FLAGS) -o $(PROG) $(OBJECTS) $(LIBS)

clean:
	rm -rf *.o $(PROG)

