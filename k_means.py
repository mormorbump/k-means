import numpy as np
import numpy.random as rd
from collections import Counter
from multivariate_gauss_sampling import MultivariateGaussSampling
import matplot_config as mat


class Kmeans:

    def __init__(self, data, K=3, c=["r", "g", "b"]):
        self.data = data
        self.N = len(data)
        self.K = K
        self.c = c
        self.mu = self.__init_mu()
        self.r = np.zeros(self.N)   # rは、どのデータがどのクラスタに所属しているかを表す指示変数

    def __init_mu(self):
        # initialize self.mu
        max_x, min_x = np.max(self.data[:, 0]), np.min(self.data[:, 0])  # 全データのx座標からmaxとminを取得
        max_y, min_y = np.max(self.data[:, 1]), np.min(self.data[:, 1])  # 全データのy座標からmaxとminを取得
        mu = np.c_[
                rd.uniform(low=min_x, high=max_x, size=self.K),
                rd.uniform(low=min_y, high=max_y, size=self.K)
            ]  # 重心(平均)μの初期値を決定。uniform でlow以上high以下の乱数をsizeぶん作成。横に繋げるので、np.c_で結合
        print("init self.mu:\n", mu)
        return mu

    def init_data_plot(self):
        # 元データをチェック
        mat.plt.figure(figsize=(12, 8))
        #  scatter(x, y, s(size), c(color), alpha(色の濃さ), marker(印の形)
        mat.plt.scatter(self.data[:, 0], self.data[:, 1], s=30, c="gray", alpha=0.5, marker="+")

        # 重心self.muの座標位置も取得し、グラフにプロット
        for i in range(self.K):
            mat.plt.scatter([self.mu[i, 0]], [self.mu[i, 1]], c=self.c[i], marker="o")

        mat.plt.title("initial state")
        mat.plt.show()

    def k_means_algorithm(self):
        """
        k-means_GMM algorithm
        """
        # 重心を最適化するため、適当な回数繰り返し処理
        for _iter in range(100):
            self.__step1()
            self.__step2()
            diff = self.mu - self.mu_prev  # 差分を取得。
            print(_iter , "diff:\n", diff)
            # self.__plot(_iter)

            if np.abs(np.sum(diff)) < 10e-4:
                print("mu is conberged.")
                break

    def __step1(self):
        """
        Step1
        クラスタを固定し、データと最も近いクラスタが対応するようrを決定。
        """

        # 各データとクラスタそれぞれのノルム（距離）を出し、その距離が最小となるクラスタにデータを所属させるゾーン
        for i in range(self.N):
            # numpy.linalg.normで、ベクトルのノルム(大きさ)を求められる。
            # クラスタの個数ぶん繰り返し処理し、データ一つに対しそれぞれのクラスタとの差を大きさそして求め、配列化。
            # その最小値のindex（つまりそれがクラスタを表す）をrに代入。これでrは「dataのi番目が、どのクラスタに所属するか」という情報を持つ。
            self.__r_update(i)

    def __r_update(self, i):
        self.r[i] = np.argmin([np.linalg.norm(self.data[i] - self.mu[k]) for k in range(self.K)])

    def __step2(self):
        """
        Step2
        Step1で求めたr(データがどのクラスタに所属しているか)を元に、重心μを更新。
        """
        cnt = dict(Counter(self.r))  # 各クラスタに所属しているデータの個数を辞書に保存。
        N_k = [cnt[k] for k in range(self.K)]  # 各クラスタのデータ数を配列に変換
        self.mu_prev = self.mu.copy()  # 更新前の重心をself.mu_prevとしてコピー
        # μの更新式より。
        # np.sum(self.data[r == k], axis=0) で「クラスタに所属しているデータの座標の合計」を取得
        # N_k[k]には各クラスタのデータ数が保存されているので、それで割ることで平均を取得。
        # これがrを固定した時に損失関数を各μで偏微分した値、その微分関数が0となるようなμの値となる。
        # したがって、重心座標をこの値に更新したときが現在のクラスタに所属するデータにおいての平均値である。
        self.__mu_update(N_k)

    def __mu_update(self, N_k):
        self.mu = np.asanyarray(
            (
                [np.sum(self.data[self.r == k], axis=0) / N_k[k] for k in range(self.K)]
            )
        )

    def __plot(self, _iter):
        #図示
        mat.plt.figure(figsize=(12, 8))
        for i in range(self.N):
            self.__plot_data_scatter(i)

        for i in range(self.K):
            self.__plot_add_arrow(i)

        mat.plt.title("iter:{}".format(_iter))
        mat.plt.show()

    def __plot_data_scatter(self, i):
        mat.plt.scatter(self.data[i, 0], self.data[i, 1], s=30, c=self.c[int(self.r[i])], alpha=0.5, marker="+")

    def __plot_add_arrow(self, i):
        ax = mat.plt.axes()
        # ax.arrowで矢印を図示できる。https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.arrow.html
        ax.arrow(
            self.mu_prev[i, 0], self.mu_prev[i, 1],  # xの始点座標, yの始点座標
            self.mu[i, 0] - self.mu_prev[i, 0], self.mu[i, 1] - self.mu_prev[i, 1],
            # xの終点座標, yの終点座標, x方向の矢の長さ, y方向の矢の長さ
            lw=0.8, head_width=0.02, head_length=0.02, fc="k", ec="k"
            # lw:線の太さ, head_width:矢の部分の幅, head_length: 矢の部分のながさ, fc: facecolor(色), ec:edgecolor(色)
        )
        mat.plt.scatter([self.mu_prev[i, 0]], [self.mu_prev[i, 1]], c=self.c[i], marker="o", alpha=0.8)  # 更新前の重心座標
        mat.plt.scatter([self.mu[i, 0]], [self.mu[i, 1]], c=self.c[i], marker="o", edgecolors="k",
                        linewidths=1)  # 更新後の重心座標


if __name__ == "__main__":
    multivariate_gauss_sampling = MultivariateGaussSampling()
    data = multivariate_gauss_sampling.copy_org_data()
    k_means = Kmeans(data)
    k_means.k_means_algorithm()
    # k_means.init_data_plot()