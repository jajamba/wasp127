# -*- coding: utf-8 -*-
import numpy as np
import wasp127_machine
import sys

print(sys.version)


bin_starts = np.array([ 1.12,  1.18,  1.24,  1.3 ,  1.36,  1.42,  1.48,  1.54,  1.6 ])
bin_ends = np.array([ 1.18,  1.24,  1.3 ,  1.36,  1.42,  1.48,  1.54,  1.6 ,  1.66])


for i in range(1,len(bin_starts)):
    bin = [bin_starts[i], bin_ends[i]]
    print(bin)
    wasp127_machine.main(wlrange=bin)
