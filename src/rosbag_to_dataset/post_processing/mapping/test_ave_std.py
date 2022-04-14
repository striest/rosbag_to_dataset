# import numpy as np
# from math import sqrt

# aa = np.random.rand(100)
# aa_mean = aa.mean()
# aa_std = aa.std()

# mean = 0
# meansq = 0
# for k, aaa in enumerate(aa):
#     mean = float(k)/(k+1)*mean + aaa/(k+1)
#     meansq = float(k)/(k+1)*meansq + aaa*aaa/(k+1)
#     var = meansq - mean*mean
#     std = sqrt(var)

# print(aa_mean, aa_std)
# print(mean, std)

from multiprocessing import Pool

def f(x):
    return x*x

if __name__ == '__main__':
    with Pool(5) as p:
        res = (p.map(f, [1, 2, 3]))
        print (res)