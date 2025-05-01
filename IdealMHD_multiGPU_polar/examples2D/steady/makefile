CC = nvcc
CFLAGS  = -ccbin mpic++ \
          -gencode=arch=compute_60,code=sm_80 \
		  --extended-lambda \
          -rdc=true \
          -Xcompiler "-Wall -O3"
SRCDIRS = ../../IdealMHD_multiGPU_polar2D
SOURCES = $(foreach dir,$(SRCDIRS),$(wildcard $(dir)/*.cu))
OBJS = $(patsubst %.cu,%.o,$(SOURCES))
MAIN = main_steady.o
EXECUTABLE = program

.PHONY: all clean

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJS) $(MAIN)
	$(CC) $(CFLAGS) $(OBJS) $(MAIN) -o $(EXECUTABLE) 

%.o: %.cu
	$(CC) $(CFLAGS) -c $< -o $@

$(MAIN): main_steady.cu
	$(CC) $(CFLAGS) -c $< -o $@

run: $(EXECUTABLE)
	./$(EXECUTABLE)

clean:
	rm -f $(OBJS) $(EXECUTABLE) $(MAIN)
