import numpy as np
import matplotlib.pyplot as plt
from tpllg.montecarlo import Point, SerieLineaire


X1 = Point(1, .1)
X2 = Point(2, .3)
Y = X1+X2
Y.show()
Y = X1.apply_func(np.exp)
Y.show()


X = np.linspace(0, 10, 10)
Y = 2*X+1
Xexp = X + np.random.normal(0, .2, 10)
Yexp = Y + np.random.normal(0, .5, 10)
S = SerieLineaire(Xexp, .2, Yexp, .5)
a, b = S.ajuste()
plt.figure()
plt.plot(S.xi, S.yi, 'o')
plt.plot(X, a.val*X+b.val)
a.show()
b.show()
plt.show()
