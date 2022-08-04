#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import array
import semopy as sem
import pingouin as pg

from tabulate import tabulate
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo


if __name__ == "__main__":

    infile = "../data/feedback/pre2.csv"

    df = pd.read_csv(infile, delimiter="\t")
    df.dropna(inplace=True)

    # Bartlett’s test 
    chi_square_value,p_value = calculate_bartlett_sphericity(df)
    print("\nChi^2 value:{}, p-value {}".format(chi_square_value, p_value))

    # Kaiser-Meyer-Olkin (KMO) Test
    kmo_all, kmo_model = calculate_kmo(df)
    print("Kaiser-Meyer-Olkin test: {}".format(kmo_model))

    # print("Cronbach alpha: {}\n".format(pg.cronbach_alpha(data=df)))



    fa = FactorAnalyzer()
    fa.fit(df)
    ev, v = fa.get_eigenvalues()
    print("Eigenvalues")
    print(ev)

    # # Create scree plot using matplotlib
    # plt.scatter(range(1,df.shape[1]+1),ev)
    # plt.plot(range(1,df.shape[1]+1),ev)
    # plt.title('Scree Plot')
    # plt.xlabel('Factors')
    # plt.ylabel('Eigenvalue')
    # plt.grid()
    # plt.show()

    n_fact = len([x for x in ev if x > 1])
    n_fact = 2

    print("\nFactor(s) > 1: {}".format(n_fact))

    fa = FactorAnalyzer(n_factors=n_fact)
    fa.set_params(rotation='varimax')
    fa.fit(df)

    float_formatter = "{:.2f}".format
    np.set_printoptions(formatter={'float_kind':float_formatter})
    
    print("\nLoadings")
    print(fa.loadings_)

    print()
    print(fa.get_communalities())

    tab = [np.append([x], y) for x,y in zip(['SS Loadings', 'Proportion Var', 'Cumulative Var'], fa.get_factor_variance())]
    print()
    print(tabulate(tab, headers=[""]+["factor {}".format(x) for x in range(1,n_fact+1)], tablefmt="github"))



    # mod = """
    # # measurement model
    # factor1 =~ y1 + y4 + y6 + y13 + y14 + y16
    # factor2 =~ y2 + y3 + y9 + y10 + y15
    # # regressions
    # factor1 ~ x1
    # factor2 ~ x1
    # factor2 ~ factor1
    # """

    # model = sem.Model(mod)
    # model.fit(df, obj="MLW", solver="SLSQP")#, groups=['x1'])
    # print(model.inspect(mode='list', what="names", std_est=True))

    # g = sem.semplot(model, "model.png")
    # print(sem.calc_stats(model))


