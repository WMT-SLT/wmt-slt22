#!/usr/bin/env python3

import sys

prefix = sys.argv[1]
a, x, b, y = [int(v) for v in sys.argv[2:6]]

i = 1
for n in [a]*x + [b]*y:
    j = i+n-1
    print("{}\t{}\t{}".format(prefix, i, j))
    i = j+1
