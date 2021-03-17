#!/bin/bash

OMP_NUM_THREADS=1 taskset -c 0 python3 main.py -name w0 -config_file topo.yaml&
OMP_NUM_THREADS=1 taskset -c 0 python3 main.py -name w1 -config_file topo.yaml&
OMP_NUM_THREADS=1 taskset -c 0 python3 main.py -name w2 -config_file topo.yaml&
OMP_NUM_THREADS=1 taskset -c 0 python3 main.py -name w3 -config_file topo.yaml&



