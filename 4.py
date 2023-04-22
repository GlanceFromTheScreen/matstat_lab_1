import numpy as np
import scipy.stats as stats
from scipy.stats import norm
import matplotlib.pyplot as plt
import math as m


def mean(data):
    return np.mean(data)

def dispersion_exp(sample):
    return mean(list(map(lambda x: x*x, sample))) \
           - (mean(sample))**2

def normal(size):
    return np.random.standard_normal(size=size)


def task4(x_set : list, n_set : list):
    alpha = 0.05
    m_all = list()
    s_all = list()
    for i in range(len(n_set)):
        n = n_set[i]
        x = x_set[i]

        m = mean(x)
        s = np.sqrt(dispersion_exp(x))

        #  доверительные интервалы [thete1, theta2]
        m1 = [m - s*(stats.t.ppf(1-alpha/2, n-1))/np.sqrt(n-1), m + s*(stats.t.ppf(1-alpha/2, n-1))/np.sqrt(n-1)]
        s1 = [s*np.sqrt(n)/np.sqrt(stats.chi2.ppf(1-alpha/2, n-1)), s*np.sqrt(n)/np.sqrt(stats.chi2.ppf(alpha/2, n-1))]

        m_all.append(m1)
        s_all.append(s1)

        print("т: %i" % (n))
        print("m: %.2f, %.2f" % (m1[0], m1[1]))
        print("sigma: %.2f, %.2f" % (s1[0],s1[1]))

    draw_hist(x_set, m_all, s_all)
    draw_result(x_set, m_all, s_all)
    return



def task4_asymp(x_set : list, n_set : list):
    alpha = 0.05
    m_all = list()
    s_all = list()
    for i in range(len(n_set)):
        n = n_set[i]
        x = x_set[i]

        m = mean(x)
        s = np.sqrt(dispersion_exp(x))

        m_as = [m - stats.norm.ppf(1-alpha / 2)/np.sqrt(n), m + stats.norm.ppf(1 - alpha / 2)/np.sqrt(n)]
        e = (sum(list(map(lambda el: (el-m)**4, x)))/n)/s**4 - 3
        s_as = [s/np.sqrt(1+stats.norm.ppf(1-alpha / 2)*np.sqrt((e+2)/n)), s/np.sqrt(1-stats.norm.ppf(1-alpha / 2)*np.sqrt((e+2)/n))]

        m_all.append(m_as)
        s_all.append(s_as)

        print("m asymptotic :%.2f, %.2f" % (m_as[0], m_as[1]))
        print("sigma asymptotic: %.2f, %.2f" % (s_as[0], s_as[1]))
    draw_result(x_set, m_all, s_all)
    return



def draw_result(x_set : list, m_all : float, s_all : list):
    fig, (ax3, ax4) = plt.subplots(1, 2)

    # draw intervals of m
    ax3.set_ylim(0.9, 1.4)
    ax3.plot(m_all[0], [1, 1], 'ro-', label='"m" interval n = 20')
    ax3.plot(m_all[1], [1.1, 1.1], 'bo-', label='"m" interval n = 100')
    ax3.legend()

    # draw intervals of sigma
    ax4.set_ylim(0.9, 1.4)
    ax4.plot(s_all[0], [1, 1], 'ro-', label='sigma interval n = 20')
    ax4.plot(s_all[1], [1.1, 1.1], 'bo-', label='sigma interval n = 100')
    ax4.legend()

    plt.savefig('picture' + '.jpg', format='jpg')
    plt.show()
    return


def draw_hist(x_set : list, m_all : float, s_all : list):
    fig, (ax1, ax2) = plt.subplots(1, 2)

    # draw hystograms
    ax1.set_ylim(0, 1)
    ax1.hist(x_set[0], density=True, histtype='stepfilled', alpha=0.2, label='N(0, 1) hyst n=20')
    ax1.legend(loc='best', frameon=True)
    ax2.set_ylim(0, 1)
    ax2.hist(x_set[1], density=True, histtype='stepfilled', alpha=0.2, label='N(0, 1) hyst n=100')
    ax2.legend(loc='best', frameon=True)

    plt.savefig('picture' + '.jpg', format='jpg')
    plt.show()
    return


def task4_builder():
    n_set = [20, 100]
    x_20 = normal(20)
    x_100 = normal(100)
    x_set = [x_20, x_100]
    task4(x_set, n_set)
    task4_asymp(x_set, n_set)
    return


task4_builder()





