import numpy as np
import matplot_config as mat
from scipy import stats as st

# parameters
K = 3
n = 301
xx = np.linspace(-4, 7, n)  # 等差数列生成。-4~7まで301この等間隔な数列を生成。

mu = [-2, 0, 2]  # 平均
sigma = [0.5, 0.7, 1.5]  # 分散
pi = [0.2, 0.3, 0.5]  # 混合係数。ガウス分布の足し合わせが1になるように比率を調整。

# density function
pdfs = np.zeros((n, K))  # 縦:データ数、 横:クラスタ数
# クラスタ毎に正規分布を生成しpdfsにクラスタ番号と一緒に代入。
for k in range(K):
    # scipy.stats.normで正規分布のメソッドにアクセス
    # pdfはprobability density functionの略で、確率密度変数を生成
    # loc: 期待値（平均）
    # scale: 標準偏差(分散)
    # http://kaisk.hatenadiary.com/entry/2015/02/17/192955
    pdfs[:, k] = pi[k]*st.norm.pdf(xx, loc=mu[k], scale=sigma[k])


# visualization
mat.plt.figure(figsize=(14, 6))
# ガウス分布を足し合わせず、クラスタ毎にプロット。
for k in range(K):
    mat.plt.plot(xx, pdfs[:, k])
mat.plt.title("pdfs")
mat.plt.show()

mat.plt.figure(figsize=(14, 6))
# stackplotは積み上げグラフを作成できるメソッド。２番目以降の引数に足し合わせる行列を代入
mat.plt.stackplot(xx, pdfs[:, 0], pdfs[:, 1], pdfs[:, 2])
mat.plt.title("stacked")
mat.plt.show()