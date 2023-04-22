import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import statistics
from tabulate import tabulate
import matplotlib.transforms as transforms
from matplotlib.patches import Ellipse

sizes = [20, 60, 100]
ros = [0, 0.5, 0.9]
REPETITIONS = 1000


def multivariate_normal(size, ro):
    return stats.multivariate_normal.rvs([0, 0], [[1.0, ro], [ro, 1.0]], size=size)


def mix_multivariate_normal(size, ro):
    return 0.9 * stats.multivariate_normal.rvs([0, 0], [[1, 0.9], [0.9, 1]], size) + 0.1 * stats.multivariate_normal.rvs([0, 0], [[10, -0.9], [-0.9, 10]], size)


def quadrant_coefficient(x, y):
    size = len(x)
    med_x = np.median(x)
    med_y = np.median(y)
    n = {1: 0, 2: 0, 3: 0, 4: 0}
    for i in range(size):
        if x[i] >= med_x and y[i] >= med_y:
            n[1] += 1
        elif x[i] < med_x and y[i] >= med_y:
            n[2] += 1
        elif x[i] < med_x and y[i] < med_y:
            n[3] += 1
        elif x[i] >= med_x and y[i] < med_y:
            n[4] += 1
    return (n[1] + n[3] - n[2] - n[4]) / size


def count_coefficients(get_sample, size, ro):
    pearson, quadrant, spirman = [], [], []
    for i in range(REPETITIONS):  # 1000 раз считаем коэфф-ты корреляции
        sample = get_sample(size, ro)  # multivariate_normal or mix_multivariate_normal
        x, y = sample[:, 0], sample[:, 1]
        pearson.append(stats.pearsonr(x, y)[0])
        spirman.append(stats.spearmanr(x, y)[0])
        quadrant.append(quadrant_coefficient(x, y))
    return pearson, spirman, quadrant


def create_table(pearson, spirman, quadrant, size, ro):  # передвем массивы из 1000 элементов - коэфф-ты корреляции
    if ro != -1:
        rows = [['rho = ' + str(ro), '$r$', '$r_{S}$', '$r_{Q}$']]
    else:
        rows = [["size = " + str(size), '$r$', '$r_{S}$', '$r_{Q}$']]
    p = np.mean(pearson)
    s = np.mean(spirman)
    q = np.mean(quadrant)
    rows.append(['$E(z)$', np.around(p, decimals=3), np.around(s, decimals=3), np.around(q, decimals=3)])

    p = np.mean([pearson[k] ** 2 for k in range(REPETITIONS)])
    s = np.mean([spirman[k] ** 2 for k in range(REPETITIONS)])
    q = np.mean([quadrant[k] ** 2 for k in range(REPETITIONS)])
    rows.append(['$E(z^2)$', np.around(p, decimals=3), np.around(s, decimals=3), np.around(q, decimals=3)])

    p = statistics.variance(pearson)
    s = statistics.variance(spirman)
    q = statistics.variance(quadrant)
    rows.append(['$D(z)$', np.around(p, decimals=3), np.around(s, decimals=3), np.around(q, decimals=3)])

    return tabulate(rows, [], tablefmt="latex_raw")


def build_ellipse(x, y, ax, n_std=3.0, **kwargs):
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

    rad_x = np.sqrt(1 + pearson)
    rad_y = np.sqrt(1 - pearson)

    ellipse = Ellipse((0, 0), width=rad_x * 2, height=rad_y * 2, facecolor='none', **kwargs)

    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def show_ellipse(size):
    fig, ax = plt.subplots(1, 3)
    size_str = "n = " + str(size)
    titles = [size_str + r', $ \rho = 0$', size_str + r', $\rho = 0.5 $', size_str + r', $ \rho = 0.9$']
    for i in range(len(ros)):
        num, ro = i, ros[i]
        sample = multivariate_normal(size, ro)
        x, y = sample[:, 0], sample[:, 1]
        build_ellipse(x, y, ax[num], edgecolor='navy')
        ax[num].grid()
        ax[num].scatter(x, y, s=5)
        ax[num].set_title(titles[num])
    plt.savefig("n" + str(size) + ".jpg", format='jpg')
    plt.show()


if __name__ == '__main__':
    for size in sizes:
        for ro in ros:
            pearson, spirman, quadrant = count_coefficients(multivariate_normal, size, ro)
            print('\n' + str(size) + '\n' + str(create_table(pearson, spirman, quadrant, size, ro)))

        pearson, spearman, quadrant = count_coefficients(mix_multivariate_normal, size, 0)
        print('\n' + str(size) + '\n' + str(create_table(pearson, spirman, quadrant, size, -1)))
        show_ellipse(size)


