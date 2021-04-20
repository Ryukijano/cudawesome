#%%import numpy as np
from timeit import default_timer as timer
from numba import vectorize

@vectorize(["float32(float32, float32)"], target='gpu')
def VectorAdd(a,b,c):
        return a + b
def main():
    N=32000000 #No of elements per Array
    
    A = np.ones(N, dtype=np.float32)
    B = np.ones(N, dtype=np.float32)
    C = np.zeroes(N, dtype=np.float32)
    
    start = timer()
    VectorAdd(A, B, C)
    vectoradd_time = timer() - start
    
    print("C[:5]="+ str(C[:5]))
    print("C[-5:] = " + str(C[-5:]))
    
    print("VectorAdd took %f seconds " % vectoradd_time)
    
if__name__ == '__main__'
main()
# %%

# %%
