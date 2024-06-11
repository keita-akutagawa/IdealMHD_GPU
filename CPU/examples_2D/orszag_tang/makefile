CC = g++
CFLAGS = -g -std=c++20 -Wall -pedantic-errors -O3 -mtune=native -march=native
SRCDIR = ../../lib_IdealMHD_2D
SOURCES = $(wildcard $(SRCDIR)/*.cpp)
OBJS = $(patsubst $(SRCDIR)/%.cpp, $(SRCDIR)/%.o, $(SOURCES))
MAIN = main_orszag_tang.o
EXECUTABLE = program

.PHONY: all clean

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJS) $(MAIN)
	$(CC) $(CFLAGS) $(OBJS) $(MAIN) -o $(EXECUTABLE)

$(SRCDIR)/%.o: $(SRCDIR)/%.cpp
	$(CC) $(CFLAGS) -c $< -o $@

$(MAIN): main_orszag_tang.cpp
	$(CC) $(CFLAGS) -c $< -o $@

run: $(EXECUTABLE)
	./$(EXECUTABLE)

clean:
	rm -f $(OBJS) $(MAIN) $(EXECUTABLE)