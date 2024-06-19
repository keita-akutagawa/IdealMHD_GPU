CC = nvcc
CXXFLAGS = -std=c++17 -rdc=true -Xcompiler "/wd 4819"
SRCDIR = ../../../lib_IdealMHD_1D_GPU
SOURCES = $(wildcard $(SRCDIR)/*.cu)
OBJS = $(patsubst $(SRCDIR)/%.cu, $(SRCDIR)/%.o, $(SOURCES))
MAIN_SRC = main_shock_tube.cu
MAIN_OBJ = main_shock_tube.o
EXECUTABLE = program

.PHONY: all clean run

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJS) $(MAIN_OBJ)
	$(CC) $(OBJS) $(MAIN_OBJ) -o $(EXECUTABLE)

$(SRCDIR)/%.o: $(SRCDIR)/%.cu
	$(CC) $(CXXFLAGS) -c $< -o $@

$(MAIN_OBJ): $(MAIN_SRC)
	$(CC) $(CXXFLAGS) -c $< -o $@

run: $(EXECUTABLE)
	./$(EXECUTABLE)

clean:
	rm -f $(SRCDIR)/*.o $(MAIN_OBJ) $(EXECUTABLE)
