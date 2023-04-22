import numpy as np
from scipy import stats as stats
import matplotlib.pyplot as plt
import scipy.optimize as opt


def reference_function(x):
    return 2 + 2 * x


def reference_noisy_function(x):
    y = []
    for i in x:
        y.append(reference_function(i) + stats.norm.rvs(0, 1))
    return y


def get_mnk_parameters(x, y):
    beta_1 = (np.mean(x * y) - np.mean(x) * np.mean(y)) / (np.mean(x * x) - np.mean(x) ** 2)
    beta_0 = np.mean(y) - beta_1 * np.mean(x)
    return beta_0, beta_1


def least_modulus_method(parameters, x, y):
    alpha_0, alpha_1 = parameters
    sum = 0
    for i in range(len(x)):
        sum += abs(y[i] - alpha_0 - alpha_1 * x[i])
    return sum


def get_mnm_parameters(x, y):
    beta_0, beta_1 = get_mnk_parameters(x, y)
    result = opt.minimize(least_modulus_method, [beta_0, beta_1], args=(x, y), method='SLSQP')
    coefs = result.x
    alpha_0, alpha_1 = coefs[0], coefs[1]
    return alpha_0, alpha_1


def MNK(x, y):
    beta_0, beta_1 = get_mnk_parameters(x, y)
    print('beta_0 = ' + str(beta_0), 'beta_1 = ' + str(beta_1))
    y_new = [beta_0 + beta_1 * x_ for x_ in x]
    return y_new


def MNM(x, y):
    alpha_0, alpha_1 = get_mnm_parameters(x, y)
    print('alpha_0= ' + str(alpha_0), 'alpha_1 = ' + str(alpha_1))
    y_new = [alpha_0 + alpha_1 * x_ for x_ in x]
    return y_new


def get_distance(y_model, y_regr):
    dist_y = sum([(y_model[i] - y_regr[i])**2 for i in range(len(y_model))])
    return dist_y


def plot_linear_regression(text, x, y):
    y_mnk = MNK(x, y)
    y_mnm = MNM(x, y)
    y_dist_mnk = get_distance(y, y_mnk)
    y_dist_mnm = get_distance(y, y_mnm)
    print('mnk distance', y_dist_mnk)
    print('mnm distance', y_dist_mnm)
    plt.plot(x, reference_function(x), color='red', label='Модель')
    plt.plot(x, y_mnk, label="МНК", color='pink')
    plt.plot(x, y_mnm, label="МНМ", color='orange')
    plt.scatter(x, y, c='darkorchid', label='Выборка')
    plt.xlim([-1.8, 2])
    plt.grid()
    plt.legend()
    plt.savefig(text + '.jpg', format='jpg')
    plt.show()


x = np.arange(-1.8, 2, 0.2)
y = reference_noisy_function(x)
plot_linear_regression('NoPerturbations', x, y)


x = np.arange(-1.8, 2, 0.2)
y = reference_noisy_function(x)
y[0] += 10
y[-1] -= 10
plot_linear_regression('Perturbations', x, y)


