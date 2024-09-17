CC = nvcc
CFLAGS = -ccbin mpic++ -gencode=arch=compute_60,code=sm_80 -rdc=true -Xcompiler "-Wall -O3"
SRCDIR = ../../IdealMHD1D_GPU
SOURCES = $(wildcard $(SRCDIR)/*.cu)
OBJS = $(patsubst $(SRCDIR)/%.cu, $(SRCDIR)/%.o, $(SOURCES))
MAIN = main_shock_tube.o
EXECUTABLE = program

.PHONY: all clean

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJS) $(MAIN)
	$(CC) $(CFLAGS) $(OBJS) $(MAIN) -o $(EXECUTABLE)

$(SRCDIR)/%.o: $(SRCDIR)/%.cu
	$(CC) $(CFLAGS) -c $< -o $@

$(MAIN): main_shock_tube.cu
	$(CC) $(CFLAGS) -c $< -o $@

run: $(EXECUTABLE)
	$(EXECUTABLE)

clean:
	rm -f $(OBJS) $(MAIN) $(EXECUTABLE)
