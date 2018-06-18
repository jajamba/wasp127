# -*- coding: utf-8 -*-
import numpy as np
import wasp127_machine
import sys

print sys.version

bin_starts = np.arange(1., 1.661, 0.06)
bin_ends = np.arange(1.06, 1.73, 0.06)


for i in range(1,len(bin_starts)):
    bin = [bin_starts[i], bin_ends[i]]
    print bin
    wasp127_machine.main(wlrange=bin)
