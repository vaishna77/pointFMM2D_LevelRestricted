CC			=g++
CFLAGS		=-c -Wall -Ofast -fopenmp -ffast-math -ffinite-math-only
LDFLAGS		=-Ofast -fopenmp
SOURCES		=./testFMM2D.cpp ./FMM2DTree.hpp
KERNEL		=-DLOGR	# use -DONEOVERR, -DLOGR
HOMOG		=-DLOGHOMOG	# use -DHOMOG, -DLOGHOMOG
OBJECTS		=$(SOURCES:.cpp=.o)
EXECUTABLE	=./testFMM2D

all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
		$(CC) $(LDFLAGS) $(OBJECTS) -o $@

.cpp.o:
		$(CC) $(CFLAGS) $(KERNEL) $(HOMOG) $< -o $@

clean:
	rm a.out testFMM2D *.o
