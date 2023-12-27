all: naive preallocate save_tensor load_from_disk

.PHONY: naive
naive:
	python3 naive.py

.PHONY: preallocate
preallocate:
	python3 preallocate_on_gpu.py

.PHONY: save_tensor
save_tensor:
	python3 make_example_tensor.py

.PHONY: load_from_disk
load_from_disk:
	python3 load_from_disk.py
