all: comp run

run:
	cat  conf4.data | ./a.out
comp:
	nvcc lab4-timings.cu
mm:
	g++ memcpy.cpp
	./a.out
gen:
	python3 gen.py > conf4.data
test: gen run
