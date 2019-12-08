all: clean nvcc run repr
run:
	echo "in.data \nout.data\n3 5" | ./a.out
nvcc:
	nvcc bilin.cu
clean:
	echo ok
repr:
	python3 to_hex.py

