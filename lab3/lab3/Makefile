CC = nvcc
CFLAGS = -Werror cross-execution-space-call -lm -rdc=true
MAIN_FILE = main.cu
BENCHMARK_FILE = benchmark.cu
MATRIX_TEST_FILE = matrix-test.cu
FILES = image.cu classifier.cu kernel.cu
PROG = lab3
BENCHMARK = benchmark
TEST = test

all:
	@$(CC) $(CFLAGS) -o $(PROG) $(MAIN_FILE) $(FILES)

benchmark:
	@$(CC) $(CFLAGS) -o $(BENCHMARK) $(BENCHMARK_FILE) $(FILES)

matrix-test:
	@$(CC) $(CFLAGS) -o $(TEST) $(MATRIX_TEST_FILE) $(FILES)

clean:
	@rm -rf $(PROG) $(BENCHMARK) $(TEST)