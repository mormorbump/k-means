import numpy as np
import numpy.random as rd
from scipy import stats as st
from multivariate_gauss_sampling import MultivariateGaussSampling
import matplot_config as mat

multivariate_gauss_sampling = MultivariateGaussSampling()
data = multivariate_gauss_sampling.copy_org_data()

for nframe in range(100):
    global mu, sigma, pi  # 関数内でグローバル変数を定義するときに宣言。
    print("nframe:", nframe)
    mat.plt.clf()  # 続けてグラフを描きたいとき、以前の図の内容を消去するメソッド

    if nframe <= 3:
        print("initial state")
        mat.plt.scatter(data[:, 0], data[:, 1], s=30, c="gray", alpha=0.5, marker="+")