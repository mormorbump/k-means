import numpy as np
import numpy.random as rd
from scipy import stats as st
import matplot_config as mat


class MultivariateGaussSampling:
    """
    教師データサンプリング
    """
    line = "---------------------------"

    def __init__(self, D=2, K=3, n=[200, 150, 150], c=["r", "g", "b"]):
        self.D = D  # 次元数
        self.K = K  # クラスタ数
        self.n = n  # クラスタ毎のデータ数
        self.N = np.sum(self.n)  # 全体のデータ数
        self.c = c  # クラスタ毎に色分け
        self.mu_true = self.__get_mu()
        self.sigma_true = self.__get_sigma()
        self.org_data = self.__get_gauss_sampling()

    def __get_mu(self):
        # asanyarrayは引数に入れたものをndarrayに変換するメソッド。
        # ndarrayはコピーを作るので元は買えないがこれは変える。ndarrayのサブクラスを通る。主な違いは指定できる引数。
        # https://www.quora.com/What-is-the-difference-among-np-array-np-asarray-and-np-asanyarray-when-numpy-creates-array-from-an-array-like-object

        # ここでは重心である平均μの初期座標を定義
        mu_true = np.asanyarray(
            [[0.2, 0.5],
             [1.2, 0.5],
             [2.0, 0.5]]
        )
        return mu_true

        # D = mu_true.shape[1]  # 次元数

    def __get_sigma(self):
        # ガウス分布の形にランダムサンプリングするため
        # 分散、共分散行列σを決定。対角線上にx軸、y軸の分散、反対にxyの共分散が入る。
        #  https://mathtrain.jp/varcovmatrix
        sigma_true = np.asanyarray(
            [
                [[0.1, 0.085],
                 [0.085, 0.1]],

                [[0.1, -0.085],
                 [-0.085, 0.1]],

                [[0.1, 0.085],
                 [0.085, 0.1]]
            ]
        )
        return sigma_true

    def __get_gauss_sampling(self):
        rd.seed(71)
        org_data = None  # np.empty((np.sum(n), 3))
        # クラスタの数ぶん平均、共分散に対応した分布を生成し、結合
        for i in range(self.K):
            # linalg.detは行列式(|A|, またはdetA)を求めるメソッド。行列の対角線同士の差分を出力。
            # 多次元ガウス分布では共分散の逆行列、行列式が式に代入されるため、そのチェック
            #  https://mathtrain.jp/tahenryogauss

            # print("check", i, "μ:", self.mu_true[i], "Σ:", self.sigma_true[i], np.linalg.det(self.sigma_true[i]))

            # https://deepage.net/features/numpy-cr.html
            # np.c_とr_は配列を結合するメソッド。c_はr_の特殊な場合を抜き出したメソッド。
            # c_は結合の向きが最低の次元の方向となる。大抵横向きに結合(1次元配列は2次元方向になる)
            # 特徴としては、スライス表記で配列を作成できるのと、
            # 配列じゃないただの数値も配列として結合できること。
            org_data = self.__get_org_data_gauss_sampling(org_data, i)
        return org_data

    def __get_org_data_gauss_sampling(self, org_data, i):
        if org_data is None:
            # scipy.stats.multivariate_normal.rvsは多次元の時のガウス分布をランダムに生成(sampling)するメソッド。
            # 平均、分散、生成データ数を引数に入れる。
            # https://qiita.com/supersaiakujin/items/71540d1ecd60ced65add
            # np.ones[n[i]]*i は、iの数の値が入る。
            # これによって、scatterメソッドでプロットするときに、この値によってデータを区切って、さらに色分けしてマッピングすることができる。
            org_data = self.__init_org_data_gauss_sampling(i)
        else:
            # 2回目以降は前回のorg_dataにさらにガウス分布を結合
            org_data = self.__add_org_data_gauss_sampling(org_data, i)
        return org_data

    def __init_org_data_gauss_sampling(self, i):
        # rvsは混合ガウス分布に基づいてランダムに生成
        org_data = np.c_[
            st.multivariate_normal.rvs(mean=self.mu_true[i],
                                       cov=self.sigma_true[i],
                                       size=self.n[i]
                                       ),
            np.ones(self.n[i]) * i  # クラスタを表す
        ]
        return org_data

    def __add_org_data_gauss_sampling(self, org_data, i):
        org_data = np.r_[
            org_data,
            np.c_[st.multivariate_normal.rvs(mean=self.mu_true[i],
                                             cov=self.sigma_true[i],
                                             size=self.n[i]
                                             ),
                  np.ones(self.n[i]) * i]  # クラスタを表す
        ]
        return org_data

    def plot_data(self):
        for i in range(3):
            mat.plt.scatter(
                self.org_data[self.org_data[:, 2] == i][:, 0],
                self.org_data[self.org_data[:, 2] == i][:, 1],
                s=30,
                c=self.c[i],
                alpha=0.5
            )
        mat.plt.show()

    def copy_org_data(self):
        data = self.org_data[:, 0:2].copy()
        return data


if __name__ == "__main__":
    multivariate_gauss_sampling = MultivariateGaussSampling()
    multivariate_gauss_sampling.plot_data()
