NVCC := nvcc

test_gpc_sm: test_gpc_sm.cu
	$(NVCC) -o test_gpc_sm test_gpc_sm.cu -lcuda -Xptxas -dlcm=cg
