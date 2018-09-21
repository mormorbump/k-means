import numpy as np
from scipy import stats as st
import numpy.random as rd
from collections import Counter
from multivariate_gauss_sampling import MultivariateGaussSampling
import matplot_config as mat

seed = 77
rd.seed(seed)


class MultivariateGaussEMAlgorithm:
    def __init__(self, data, D=2, K=3, n=[200, 150, 150], c=["r", "g", "b"]):
        self.data = data
        self.D = D
        self.K = K
        self.n = n
        self.N = np.sum(n)
        self.likelihood = None
        self.gamma = None
        self.N_k = None
        self.c = c
        self.pi = self.__init_pi()
        self.mu = self.__init_mu()
        self.mu_prev = None
        self.sigma = self.__init_sigma()
        self.diff = None

    def __init_pi(self):
        """
        「どのクラスタに所属しているのか」という割合を決める混合係数πの初期値を決定
        :return:
        """
        pi = np.zeros(self.K)
        for k in range(self.K):
            if k == self.K-1:  # 最後のクラスタの時は1からpiに代入したものの合計を引いたものを代入
                pi[k] = 1 - np.sum(pi)
            else:
                pi[k] = 1 / self.K  # クラスタ数で正規化した値を代入。
        print("init pi:", pi)
        return pi

    def __init_max_min(self):
        max_x, min_x = np.max(self.data[:, 0]), np.min(self.data[:, 0])
        max_y, min_y = np.max(self.data[:, 1]), np.min(self.data[:, 1])
        return max_x, min_x, max_y, min_y

    def __init_mu(self):
        """
        平均、もとい重心の初期値を決定。値域に収まったx,yの値をクラスタ数分生成
        :return:
        """
        max_x, min_x, max_y, min_y = self.__init_max_min()
        # x,yそれぞれの値域の中でランダムにクラスタ数ぶん生成した配列をc_メソッドで結合。
        # c_は結合の向きが最低の次元の方向となるが、一次元配列同士の時は二次元方向に結合される。
        rand_x = rd.uniform(low=min_x, high=max_x, size=self.K)
        rand_y = rd.uniform(low=min_y, high=max_y, size=self.K)
        # print(rand_x)
        # print(rand_y)
        mu = np.c_[rand_x, rand_y]
        print("init mu:", mu)
        return mu

    def visualize_for_check(self):
        mat.plt.figure(figsize=(12, 8))
        mat.plt.scatter(self.data[:, 0], self.data[:, 1], s=30, c="gray", alpha=0.5, marker="+")
        for i in range(self.K):
            mat.plt.scatter([self.mu[i, 0]], [self.mu[i, 1]], c=self.c[i], marker="o")
        mat.plt.show()

    def __init_sigma(self):
        """
        共分散Σの初期値を決定。
        ガウス分布の形にランダムサンプリングするため分散、共分散行列σを決定。
        対角線上にx軸、y軸の分散、反対にxyの共分散が入る。
        :return:
        """
        sigma = np.asanyarray(
            [
                [[0.1, 0],
                 [0, 0.1]],

                [[0.1, 0],
                 [0, 0.1]],

                [[0.1, 0],
                 [0, 0.1]]
            ]
        )
        return sigma

    def calc_likelihood(self, data):
        """
        クラスタ毎の尤度を計算。
        確率密度関数に混合係数をかけたのを縦列に順番に保存して行く。
        カラムの番号がクラスタに対応
        :return:
        """
        likelihood = np.zeros((np.sum(self.n), 3))
        # print(self.pi[1]*st.multivariate_normal.pdf([0.4, 0.6], mean=self.mu[1], cov=self.sigma[1]))
        for k in range(self.K):
            likelihood[:, k] = [self.pi[k]*st.multivariate_normal.pdf(d, self.mu[k], self.sigma[k]) for d in data]
        print('initial sum of log likelihood:', np.sum(np.log(likelihood)))
        return likelihood

    def calc_prob_gmm(self, data):
        """
        混合ガウスの確率密度分布に従った配列を作成。
        確率密度分布とは、「ある値をxとした時に、その値となる確率をy」とした関数
        確率密度分布はpdfと表す。
        :return:
        """
        return [[self.pi[k]*st.multivariate_normal.pdf(d, self.mu[k], self.sigma[k]) for k in range(self.K)] for d in data]

    def em_algorithm(self):
        for step in range(100):
            self.__e_step()
            self.__m_step()
            self.__result_visualize(step)
            if self.__likelihood_converged():
                break

    def __e_step(self):
        self.__out_gamma()
        self.__cluster_datas_sum()

    def __out_gamma(self):
        """
        負担率の導出
        (混合係数毎のガウス分布/混合ガウス分布)の分子の部分を導出
        :return:
        """
        self.likelihood = self.calc_likelihood(self.data)  # 混合係数一つ一つのガウス分布の尤度を導出
        # gamma = np.apply_along_axis(lambda x: [xx/np.sum(x) for xx in x] , 1, likelihood)
        self.gamma = (self.likelihood.T / np.sum(self.likelihood, axis=1)).T  # 負担率（潜在変数zの事後確率）の導出

    def __cluster_datas_sum(self):
        self.N_k = [np.sum(self.gamma[:, k]) for k in range(self.K)]

    def __m_step(self):
        self.__calc_pi()
        self.__calc_mu()
        self.__calc_sigma()
        self.__calc_likelihood()  # 未実装

    def __calc_pi(self):
        """
        混合係数の更新。
        データの総量に対し、クラスタの割合がどれくらいかなので、Nで割って正規化
        :return:
        """
        self.pi = self.N_k / self.N

    def __calc_mu(self):
        """
        μの更新。
        μ_k = (Σ_N γ(z_n_k) * x_n) / N_k
        :return:
        """
        tmp_mu = np.zeros((self.K, self.D))

        for k in range(self.K):  # クラスタ毎に更新
            for n in range(self.N):  # N回分のシグマを回す
                tmp_mu[k] += self.gamma[n, k] * self.data[n]  # γ(z_k=1|x) * x_i
            tmp_mu[k] = tmp_mu[k] / self.N_k[k]  # * 1/N_k
            # print('updated mu[{}]:\n'.format(k) , tmp_mu[k])
        self.mu_prev = self.mu.copy()
        self.mu = tmp_mu.copy()
        # print('updated mu:\n', mu)

    def __calc_sigma(self):
        """
        Σ(共分散)の更新
        Σ = (Σ_N γ(z_n_k) * (x_n - μ_k) * (x_n - μ_k).T) / N_k
        (x_n - μ_k) * (x_n - μ_k).Tの部分は内積 => dot((x_n - μ_k), (x_n - μ_k).T)
        :return:
        """
        tmp_sigma = np.zeros((self.K, self.D, self.D))

        for k in range(self.K):
            tmp_sigma[k] = np.zeros((self.D, self.D))
            for i in range(self.N):
                tmp = np.asanyarray(self.data[i] - self.mu[k])[:, np.newaxis]  # axis=1のところに次元を追加
                tmp_sigma[k] += self.gamma[i, k] * np.dot(tmp, tmp.T)
            tmp_sigma[k] /= self.N_k[k]
            # print('updated sigma[{}]:\n'.format(k) , tmp_sigma[k])
        self.sigma = tmp_sigma.copy()

    def __calc_likelihood(self):
        prev_likelihood = self.likelihood
        self.likelihood = self.calc_likelihood(self.data)

        prev_sum_log_likelihood = np.sum(np.log(prev_likelihood))
        sum_log_likelihood = np.sum(np.log(self.likelihood))
        self.diff = prev_sum_log_likelihood - sum_log_likelihood

        self.__print_result(sum_log_likelihood)

    def __print_result(self, sum_log_likelihood):
        print("sum of log likelihood:", sum_log_likelihood)
        print("diff:", self.diff)
        print("pi:", self.pi)
        print("mu:", self.mu)
        print("sigma:", self.sigma)

    def __result_visualize(self, step):
        mat.plt.figure(figsize=(12, 8))
        for i in range(self.N):
            mat.plt.scatter(self.data[i, 0], self.data[i, 1], s=30, c=self.gamma[i], alpha=0.5, marker="+")

        for i in range(self.K):
            ax = mat.plt.axes()
            ax.arrow(self.mu_prev[i, 0], self.mu_prev[i, 1],
                     self.mu[i, 0]-self.mu_prev[i, 0], self.mu[i, 1]-self.mu_prev[i, 1],
                     lw=0.8, head_width=0.02, head_length=0.02, fc="k", ec="k")
            mat.plt.scatter([self.mu_prev[i, 0]], [self.mu_prev[i, 1]], c=self.c[i], marker="o", alpha=0.8)
            mat.plt.scatter([self.mu[i, 0]], [self.mu[i, 1]], c=self.c[i], marker="o", edgecolors="k", linewidths=1)
        mat.plt.title("step:{}".format(step))

        self.__print_gmm_contour()
        mat.plt.show()

    def __print_gmm_contour(self):
        """
        混合ガウス分布の確率密度分布を等高線にプロット
        display predicted scores by the model as a contour plot
        """
        from matplotlib.colors import LogNorm
        max_x, min_x, max_y, min_y = self.__init_max_min()
        X, Y = np.meshgrid(np.linspace(min_x, max_x), np.linspace(min_y, max_y))
        XX = np.array([X.ravel(), Y.ravel()]).T
        Z = np.sum(np.asanyarray(self.calc_prob_gmm(XX)), axis=1)
        Z = Z.reshape(X.shape)
        CS = mat.plt.contour(X, Y, Z, alpha=0.2, zorder=-100)  # X, Yに対応したZを高さとして渡す。
        mat.plt.title("pdf contour of a GMM")  # 混合ガウス分布の確率密度分布の等高線

    def __likelihood_converged(self):
        if np.abs(self.diff) < 0.0001:
            print("likelihood is converged.")
            return True


if __name__ == "__main__":
    multivariate_gauss_sampling = MultivariateGaussSampling()
    # print(multivariate_gauss_sampling.org_data[:, :2])
    multivariate_gauss_em_algorithm = MultivariateGaussEMAlgorithm(multivariate_gauss_sampling.org_data[:, :2])
    # multivariate_gauss_em_algorithm.calc_likelihood(multivariate_gauss_em_algorithm.data)
    # multivariate_gauss_em_algorithm.visualize_for_check()
    multivariate_gauss_em_algorithm.em_algorithm()