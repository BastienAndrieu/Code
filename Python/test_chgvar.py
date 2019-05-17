import numpy
import sys
sys.path.append('/d/bandrieu/GitHub/Code/Python/')
import lib_chebyshev as cheb
from numpy.polynomial.chebyshev import chebval

import matplotlib.pyplot as plt

#####################################################
n = 8
m = 200


c = 2.0*numpy.random.rand(n) - 1.0
c = c/(numpy.arange(1,n+1)**2)

x = numpy.linspace(-1,1,m)

x01 = 2.0*numpy.random.rand(2) - 1.0
x0 = x01[0]
x1 = x01[1]

y = numpy.linspace(x0, x1,m)

a = cheb.chgvar1(c, x0, x1)

print numpy.amax(numpy.absolute( chebval(x, a) - chebval(y, c) ))


fig, ax = plt.subplots()
ax.plot(x, chebval(x, c), 'b-', lw=1)
ax.plot(y, chebval(x, a), 'r-', lw=2)
plt.show()


