import math
import numpy as np
from scipy.stats import norm, cauchy, uniform, poisson, laplace
import matplotlib.pyplot as plt
from statistics import median, mean
import pandas as pd
import seaborn as sns


def z_R(array):
    return (min(array) + max(array)) / 2


def z_P(array, p):
    """подается уже отсторированный массив"""
    if (len(array) * p) % 1 == 0:
        ind = len(array) * p
    else:
        ind = math.floor((len(array) * p)) + 1

    return array[int(ind)]


def z_Q(array):
    return (z_P(array, 0.25) + z_P(array, 0.75)) / 2


def z_Tr(array):
    result = 0
    r = math.floor(len(array) / 4)
    for i in range(r, len(array) - r - 1, 1):
        result += array[i]

    return result / (len(array) - 2 * r)


def Distribution_Graphics(distribution, loc=0, scale=1, interval=0.005):

    name = distribution.name

    #  выбираем 10 нормально распределенных точек
    dist10 = distribution.rvs(size=10, loc=loc, scale=scale)
    dist50 = distribution.rvs(size=50, loc=loc, scale=scale)
    dist1000 = distribution.rvs(size=1000, loc=loc, scale=scale)

    plt.subplot(131)

    plt.tight_layout()
    # plt.tight_layout(h_pad=0.1)

    #  устанавливаем точки по оси икс
    x = np.linspace(distribution.ppf(interval, loc, scale), distribution.ppf(1 - interval, loc, scale), 100)

    #  плотность распределения:
    ax1 = plt.subplot(1, 3, 1)
    ax1.plot(x, distribution.pdf(x, loc, scale), 'r-', lw=2, alpha=0.6, label=f'{name} pdf')

    #  строим гистограмму
    ax1.hist(dist10, density=True, bins='auto', edgecolor='black', alpha=0.5)
    ax1.legend(f'{name}')
    ax1.set_xlabel(f'n = 10, {distribution.name}Numbers')
    ax1.set_ylabel('density')
    ax1.set_ylim(0, 1)
    ax1.legend()
    ax1.set_box_aspect(1)

    ax2 = plt.subplot(1, 3, 2)
    ax2.plot(x, distribution.pdf(x, loc, scale), 'r-', lw=2, alpha=0.6, label='{} pdf'.format(name))
    ax2.hist(dist50, density=True, bins='auto', edgecolor='black', alpha=0.5)
    ax2.set_xlabel(f'n = 50, {distribution.name}Numbers')
    ax2.set_ylabel('density')
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.set_box_aspect(1)

    ax3 = plt.subplot(1, 3, 3)
    ax3.plot(x, distribution.pdf(x, loc, scale), 'r-', lw=2, alpha=0.6, label='{} pdf'.format(name))
    ax3.hist(dist1000, density=True, bins='auto', edgecolor='black', alpha=0.5)
    ax3.set_xlabel(f'n = 1000, {distribution.name}Numbers')
    ax3.set_ylabel('density')
    ax3.set_ylim(0, 1)
    ax3.legend()
    ax3.set_box_aspect(1)

    plt.show()


def Poisson_Graphics(mu=10):
    plt.subplot()

    x = np.arange(poisson.ppf(0.001, mu),
                  poisson.ppf(0.999, mu))

    ax1 = plt.subplot(1, 3, 1)
    x_hist = poisson.rvs(mu, size=10)
    ax1.hist(x_hist, density=True, edgecolor='black', alpha=0.5)
    ax1.plot(x, poisson.pmf(x, mu), 'r-', ms=8, label='poisson pmf', alpha=0.6)
    ax1.legend('poisson')
    ax1.set_xlabel('n = 10, poissonNumbers')
    ax1.set_ylabel('density')
    ax1.set_ylim(0, 1)
    ax1.legend()
    ax1.set_box_aspect(1)

    ax2 = plt.subplot(1, 3, 2)
    x_hist = poisson.rvs(mu, size=50)
    ax2.hist(x_hist, density=True, edgecolor='black', alpha=0.5)
    ax2.plot(x, poisson.pmf(x, mu), 'r-', ms=8, label='poisson pmf', alpha=0.6)
    ax2.legend('poisson')
    ax2.set_xlabel('n = 50, poissonNumbers')
    ax2.set_ylabel('density')
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.set_box_aspect(1)

    ax3 = plt.subplot(1, 3, 3)
    x_hist = poisson.rvs(mu, size=1000)
    ax3.hist(x_hist, density=True, edgecolor='black', alpha=0.5)
    ax3.plot(x, poisson.pmf(x, mu), 'r-', ms=8, label='poisson pmf', alpha=0.6)
    ax3.legend('poisson')
    ax3.set_xlabel('n = 1000, poissonNumbers')
    ax3.set_ylabel('density')
    ax3.set_ylim(0, 1)
    ax3.legend()
    ax3.set_box_aspect(1)

    plt.show()


def Characteristics_Evaluate(distribution, loc=0, scale=1):
    super_dict = {'10': {'mean': 0,
                         'median': 0,
                         'z_R': 0,
                         'z_Q': 0,
                         'z_Tr': 0},
                  '100': {'mean': 0,
                          'median': 0,
                          'z_R': 0,
                          'z_Q': 0,
                          'z_Tr': 0},
                  '1000': {'mean': 0,
                           'median': 0,
                           'z_R': 0,
                           'z_Q': 0,
                           'z_Tr': 0}}

    distributions_dict = {'10': [], '100': [], '1000': []}

    super_dict_power_two = {'10': super_dict['10'].copy(),
                            '100': super_dict['100'].copy(),
                            '1000': super_dict['1000'].copy()}

    commands = [mean, median, z_R, z_Q, z_Tr]

    for i in range(1000):
        distributions_dict['10'] = distribution.rvs(size=10, loc=loc, scale=scale)
        distributions_dict['100'] = distribution.rvs(size=100, loc=loc, scale=scale)
        distributions_dict['1000'] = distribution.rvs(size=1000, loc=loc, scale=scale)

        distributions_dict['10'].sort()
        distributions_dict['100'].sort()
        distributions_dict['1000'].sort()

        # print(mean(distributions_dict['1000']))

        if i == 0:
            for cmd in commands:
                for key in super_dict.keys():
                    super_dict[key][cmd.__name__] = cmd(distributions_dict[key])
                    super_dict_power_two[key][cmd.__name__] = cmd(distributions_dict[key]) ** 2

        else:
            for cmd in commands:
                for key in super_dict:
                    super_dict[key][cmd.__name__] = (super_dict[key][cmd.__name__] * i + cmd(distributions_dict[key])) / (i + 1)
                    super_dict_power_two[key][cmd.__name__] = ((super_dict_power_two[key][cmd.__name__] * i +
                                                               cmd(distributions_dict[key]) ** 2) /
                                                               (i + 1))

    print(super_dict)
    return super_dict_power_two


def Poisson_Evaluate(mu):
    super_dict = {'10': {'mean': 0,
                         'median': 0,
                         'z_R': 0,
                         'z_Q': 0,
                         'z_Tr': 0},
                  '100': {'mean': 0,
                          'median': 0,
                          'z_R': 0,
                          'z_Q': 0,
                          'z_Tr': 0},
                  '1000': {'mean': 0,
                           'median': 0,
                           'z_R': 0,
                           'z_Q': 0,
                           'z_Tr': 0}}
    commands = [mean, median, z_R, z_Q, z_Tr]
    distributions_dict = {'10': [], '100': [], '1000': []}

    super_dict_power_two = {'10': super_dict['10'].copy(),
                            '100': super_dict['100'].copy(),
                            '1000': super_dict['1000'].copy()}

    for i in range(1000):
        distributions_dict['10'] = [float(x) for x in poisson.rvs(mu, size=10)]
        distributions_dict['100'] = [float(x) for x in poisson.rvs(mu, size=100)]
        distributions_dict['1000'] = [float(x) for x in poisson.rvs(mu, size=1000)]

        # print(mean([float(x) for x in distributions_dict['1000']]))

        distributions_dict['10'].sort()
        distributions_dict['100'].sort()
        distributions_dict['1000'].sort()

        if i == 0:
            for cmd in commands:
                for key in super_dict.keys():
                    super_dict[key][cmd.__name__] = cmd(distributions_dict[key])
                    super_dict_power_two[key][cmd.__name__] = cmd(distributions_dict[key]) ** 2

        else:
            for cmd in commands:
                for key in super_dict:
                    super_dict[key][cmd.__name__] = (super_dict[key][cmd.__name__] * i + cmd(distributions_dict[key])) / (
                                i + 1)
                    super_dict_power_two[key][cmd.__name__] = ((super_dict_power_two[key][cmd.__name__] * i +
                                                               cmd(distributions_dict[key]) ** 2) /
                                                               (i + 1))
    return super_dict_power_two


def Box_Plot(distribution, loc=0, scale=1):

    dist20 = distribution.rvs(size=20, loc=loc, scale=scale)
    dist100 = distribution.rvs(size=100, loc=loc, scale=scale)

    plt.subplot(211)
    plt.title(f'{distribution.name} distribution')

    plt.subplot(2, 1, 1)
    sns.boxplot(x=dist20)
    plt.xlabel('x')
    plt.ylabel('20')
    x1, x2 = plt.xlim()

    plt.subplot(2, 1, 2)
    sns.boxplot(x=dist100)
    plt.xlabel('x')
    plt.ylabel('100')
    plt.xlim(x1, x2)

    plt.show()


def Poisson_Box_Plot(mu):
    dist20 = poisson.rvs(mu, size=20)
    dist100 = poisson.rvs(mu, size=100)

    plt.subplot(211)
    plt.title('poisson distribution')



    plt.subplot(2, 1, 2)
    sns.boxplot(x=dist100)
    plt.xlabel('x')
    plt.ylabel('100')
    x1, x2 = plt.xlim()


    plt.subplot(2, 1, 1)
    sns.boxplot(x=dist20)
    plt.xlabel('x')
    plt.ylabel('20')
    plt.xlim(x1, x2)


    plt.show()


def Emissions(distribution, loc=0, scale=1, dist_size=20):

    emissions_part = 0

    for i in range(1000):
        dist = distribution.rvs(size=dist_size, loc=loc, scale=scale)
        dist.sort()

        z_1_4 = z_P(dist, 0.25)
        z_3_4 = z_P(dist, 0.75)

        x_1_20 = z_1_4 - 3 * (z_3_4 - z_1_4) / 2
        x_2_20 = z_3_4 + 3 * (z_3_4 - z_1_4) / 2

        emissions = [x for x in dist if (x < x_1_20) or (x > x_2_20)]
        emissions_part += len(emissions) / len(dist)
    print(distribution.name, dist_size, emissions_part / 1000)
    # print(emissions_part / 1000)


def Poisson_Emissions(mu, dist_size=20):

    emissions_part = 0

    for i in range(1000):
        dist = [float(x) for x in poisson.rvs(mu, size=dist_size)]
        dist.sort()

        z_1_4 = z_P(dist, 0.25)
        z_3_4 = z_P(dist, 0.75)

        x_1_20 = z_1_4 - 3 * (z_3_4 - z_1_4) / 2
        x_2_20 = z_3_4 + 3 * (z_3_4 - z_1_4) / 2

        emissions = [x for x in dist if (x < x_1_20) or (x > x_2_20)]
        emissions_part += len(emissions) / len(dist)
    print(emissions_part / 1000)


def Kernel(distribution, loc=0, scale=1, dist_size=20):

    h = 1.06 * (scale ** 0.5) * (dist_size ** (-0.2))

    dist = distribution.rvs(size=dist_size, loc=loc, scale=scale)
    x = np.linspace(-4, 4, 100)

    plt.subplot(311)
    for coef in [0.5, 1, 2]:

        ax = plt.subplot(1, 3, int(coef // 1 + 1))
        sns.kdeplot(data=dist,
                    bw_adjust=coef*h,
                    bw_method='silverman')
        plt.plot(x, distribution.pdf(x, loc, scale), 'g-', lw=2, alpha=0.8, label='pdf')
        plt.ylim(0, 1)
        plt.xlim(-4, 4)
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title(f'h = h_n * {coef}, n = {dist_size}')
        ax.set_box_aspect(1)

    plt.show()


def Poisson_Kernel(mu, dist_size=20):

    scale = mu
    h = 1.06 * (scale ** 0.5) * (dist_size ** (-0.2))

    dist = [float(x) for x in poisson.rvs(mu, size=dist_size)]
    x = np.arange(6, 15)

    plt.subplot(311)
    for coef in [0.5, 1, 2]:

        plt.subplot(1, 3, int(coef // 1 + 1))
        sns.kdeplot(bw_adjust=0.5*coef*h,
                    label='kernel',
                    data=dist)
        plt.plot(x, poisson.pmf(x, mu), 'g-', ms=8, label='poisson pmf')
        plt.ylim(0, 1)
        plt.xlim(6, 14)
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title(f'h = h_n * {coef}, n = {dist_size}')
        plt.legend()

    plt.show()


def write_in_file(super_dict):
    df = pd.DataFrame.from_dict(super_dict, 'index').reset_index()
    return df


def write_concat_df_in_file():
    frames = []
    distr_array = [[norm, 0, 1], [cauchy, 0, 1], [laplace, 0, 1 / math.sqrt(2)], [uniform, -(3 ** 0.5), (3 ** 0.5)]]
    count = 0
    for sub_arr in distr_array:
        count+=4
        sup_dict = write_in_file(super_dict=Characteristics_Evaluate(sub_arr[0], sub_arr[1], sub_arr[2]))
        df2 = pd.DataFrame(np.insert(sup_dict.values, 0, values=[f'{sub_arr[0]}', '', '', '', '', ''], axis=0))
        df2.columns = sup_dict.columns
        frames.append(df2)

    poisson1 = Poisson_Evaluate(10)
    sup_dict = write_in_file(super_dict=poisson1)
    df2 = pd.DataFrame(np.insert(sup_dict.values, 0, values=['poisson', '', '', '', '', ''], axis=0))
    df2.columns = sup_dict.columns
    frames.append(df2)

    result = pd.concat(frames)
    result.to_excel('fole.xlsx', index=False)


def Emissions_All():
    for size in [20, 100]:
        Emissions(norm, dist_size=size)
        Emissions(cauchy, dist_size=size)
        Emissions(laplace, loc=0, scale=1 / (2**0.5), dist_size=size)
        Emissions(uniform, loc=-(3 ** 0.5), scale=2 * (3 ** 0.5), dist_size=size)
        Poisson_Emissions(mu=10, dist_size=size)


if __name__ == '__main__':
    # Distribution_Graphics(uniform, -(3**0.5), 2*(3**0.5))
    # Poisson_Graphics(10)
    # Characteristics_Evaluate(norm)
    # Poisson_Evaluate(10)
    # write_concat_df_in_file()
    # Box_Plot(uniform, loc=-(3 ** 0.5), scale=2 * (3 ** 0.5))
    # Poisson_Box_Plot(10)
    # Emissions_All()
    # Kernel(uniform, loc=-(3 ** 0.5), scale=2 * (3 ** 0.5), dist_size=100)
    Poisson_Kernel(10, dist_size=100)



