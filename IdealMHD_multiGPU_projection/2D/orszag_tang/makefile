CC = nvcc
CFLAGS  = -ccbin mpic++ \
          -I$(AMGX_ROOT)/include \
          -L$(AMGX_ROOT)/lib \
          -gencode=arch=compute_60,code=sm_80 \
          -rdc=true \
          -Xcompiler "-Wall -O3"
SRCDIR = ../IdealMHD2D_GPU
SOURCES = $(wildcard $(SRCDIR)/*.cu)
OBJS = $(patsubst $(SRCDIR)/%.cu, $(SRCDIR)/%.o, $(SOURCES))
MAIN = main_orszag_tang.o
EXECUTABLE = program

.PHONY: all clean

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJS) $(MAIN)
	$(CC) $(CFLAGS) $(OBJS) $(MAIN) -o $(EXECUTABLE) \
		-lamgxsh -lcusparse -lcublas -lcudart -ldl -lmpi \
		-Xlinker -rpath -Xlinker $(AMGX_ROOT)/lib


$(SRCDIR)/%.o: $(SRCDIR)/%.cu
	$(CC) $(CFLAGS) -c $< -o $@

$(MAIN): main_orszag_tang.cu
	$(CC) $(CFLAGS) -c $< -o $@

run: $(EXECUTABLE)
	$(EXECUTABLE)

clean:
	rm -f $(OBJS) $(MAIN) $(EXECUTABLE)