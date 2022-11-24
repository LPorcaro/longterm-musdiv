#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import array
import semopy as sem
import pingouin as pg
import csv
import math
import collections

from tabulate import tabulate
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo

TITLES = ["I listen to Electronic Music\nonly for partying.",
          "I believe Electronic Music\nhas mostly fast tempo\nand high energy.",
          "I think of Electronic Music\nas a varied genre with\nseveral sub-styles.",
          "I believe that Electronic\nMusic could fit in\ndifferent contexts."]



def get_pids():
    """
    """

    HD, LD = [], []

    with open("../data/HD_PID.csv", 'r') as inf:
        _reader = csv.reader(inf)
        for row in _reader:
            HD.append(row[0])

    with open("../data/LD_PID.csv", 'r') as inf:
        _reader = csv.reader(inf)
        for row in _reader:
            LD.append(row[0])

    return HD, LD

def norm_dict(d):
    """
    """

    factor = 1.0/sum(d.values())
    for k in d:
        d[k] = round(d[k]*factor,2)

    for i in range(1, 6):
        if i not in d:
            d[i] = 0 

    return d

def stats_sum(df):
    """
    """
    return df.median(), round(df.mean(), 2), round(df.std(), 2)


def weight_mean_dict(d: dict) -> float:
    total = 0
    count = 0
    for k, v in d.items():
        total += k * v
        count += v

    mean = total / count
    return mean


def mwu_groups():
    """
    """
    # MWU compare HD/LD
    for col in df_HD.columns[1:]:
        print("\n#######",col)
        d_HD = norm_dict(df_HD[col].value_counts(sort=False).to_dict())
        d_LD = norm_dict(df_LD[col].value_counts(sort=False).to_dict())
        print("HD", dict(sorted(d_HD.items())), round(df_HD[col].mean(),2), round(df_HD[col].std(),2))
        print("LD", dict(sorted(d_LD.items())), round(df_LD[col].mean(),2), round(df_LD[col].std(),2))
        # print(df_HD[col].describe())
        # print(df_LD[col].describe())
        print (pg.mwu(df_HD[col], df_LD[col]))   


def plot_stereo():
    """
    """
    # Plot Coupled Questions
    q_tuples = [("q1:7", "q5:6"),("q1:8", "q5:5"), ("q2:3", "q6:5"), ("q2:4", "q6:6")]

    fig, axs = plt.subplots(2, len(q_tuples), sharey=True)
    for n, tup in enumerate(q_tuples):
        print(tup)
        q1, q2 = tup

        X_axis = np.arange(5)

        d_HD_1 = norm_dict(df_HD[q1].value_counts(sort=False).to_dict())
        d_HD_2 = norm_dict(df_HD[q2].value_counts(sort=False).to_dict())
        d_LD_1 = norm_dict(df_LD[q1].value_counts(sort=False).to_dict())
        d_LD_2 = norm_dict(df_LD[q2].value_counts(sort=False).to_dict())

        axs[0, n].grid(axis='y')
        axs[1, n].grid(axis='y')
        
        axs[0, n].bar(X_axis - 0.2, [d_HD_1[v] for v in [1,2,3,4,5]], 0.4, label = 'Before', color='brown')
        axs[0, n].bar(X_axis + 0.2, [d_HD_2[v] for v in [1,2,3,4,5]], 0.4, label = 'After', color='tomato', hatch="-")
        axs[1, n].bar(X_axis - 0.2, [d_LD_1[v] for v in [1,2,3,4,5]], 0.4, label = 'Before', color='forestgreen')
        axs[1, n].bar(X_axis + 0.2, [d_LD_2[v] for v in [1,2,3,4,5]], 0.4, label = 'After', color='limegreen', hatch="-")

        axs[0, n].set_xticks(X_axis)
        axs[0, n].set_xticklabels([1,2,3,4,5], fontsize=18)
        axs[1, n].set_xticks(X_axis)
        axs[1, n].set_xticklabels([1,2,3,4,5], fontsize=18)
        axs[0, n].set_title(TITLES[n], fontsize=20)
        axs[0, n].tick_params(axis='y', labelsize=18)
        axs[1, n].tick_params(axis='y', labelsize=18)

    axs[0,0].set_ylabel('% of responses', fontsize=20)
    axs[1,0].set_ylabel('% of responses', fontsize=20)
    axs[0,3].legend(fontsize=18,  bbox_to_anchor=(.85, 1),)
    axs[1,3].legend(fontsize=18,  bbox_to_anchor=(.85, 1),)
    plt.show()   




if __name__ == "__main__":

    infile = "../data/feedback/data.csv"
    # infile = "../data/feedback/01_pre.csv"
    # infile = "../data/feedback/02_cond.csv"
    # infile = "../data/feedback/03_post.csv"
    # infile = "../data/feedback/04_overall.csv"

    HD, LD = get_pids()

    df = pd.read_csv(infile, delimiter="\t")
    df.dropna(inplace=True)

    df_val = df.drop('PROLIFIC_PID', axis=1)


    df_HD = df[df['PROLIFIC_PID'].isin(HD)]
    df_LD = df[df['PROLIFIC_PID'].isin(LD)]
    df_val_HD = df_HD.drop('PROLIFIC_PID', axis=1)
    df_val_LD = df_LD.drop('PROLIFIC_PID', axis=1)

    # plot_stereo()

    ############ PLOT 

    ## Openess / Appreciation Items
    # PRE
    o_items_p = ["q1:1", "q1:2", "q1:3"]
    o_items_n = ["q2:5", "q2:6", "q2:7"]    
    o_items = o_items_p + o_items_n
    a_items_p = ["q1:4", "q2:1", "q2:2"]
    a_items_n = ["q2:8", "q1:5", "q1:6"]
    a_items = a_items_p + a_items_n

    # # COND
    # o_items_p = ["q3:1", "q3:2", "q3:3"]
    # o_items_n = ["q4:1", "q4:2", "q4:3"]
    # o_items = o_items_p + o_items_n
    # a_items_p = ["q3:4", "q4:5", "q4:6"]
    # a_items_n = ["q4:4", "q3:5", "q3:6"]
    # a_items = a_items_p + a_items_n   

    # # POST
    # o_items_p = ["q5:1", "q5:2"]
    # o_items_n = ["q6:1", "q6:2"]
    # o_items = o_items_p + o_items_n
    # a_items_p = ["q5:3", "q6:4"]
    # a_items_n = ["q5:4", "q6:3"]
    # a_items = a_items_p + a_items_n   


    fig, axs = plt.subplots(1, 6, sharey=True)
    ## Consistency + Distribution
    # Openness
    df_val[o_items] -= 3
    df_val[o_items_n] = -df_val[o_items_n]
    df_val[o_items] += 3
    # print("\nCronbach alpha: {}".format(pg.cronbach_alpha(data=df_val[o_items])))

    o_counts = []
    for item in o_items:
        o_counts.append(df_val[item].value_counts().to_dict())
    counter = collections.Counter()
    for d in o_counts: 
        counter.update(d)
    result = dict(counter)  

    l = []
    for (k,v) in result.items():
        l += v*[k]

    print(round(np.mean(l),2), round(np.std(l),2))
    print(norm_dict(result))

    axs[0].bar(norm_dict(result).keys(), norm_dict(result).values(), 0.5, color='blue', label='openness')

    # Appreciation
    df_val[a_items] -= 3 
    df_val[a_items_n] = -df_val[a_items_n]
    df_val[a_items] += 3 
    # print("\nCronbach alpha: {}".format(pg.cronbach_alpha(data=df_val[a_items])))

    a_counts = []
    for item in a_items:
        a_counts.append(df_val[item].value_counts().to_dict())
    counter = collections.Counter()
    for d in a_counts: 
        counter.update(d)
    result = dict(counter)  

    l = []
    for (k,v) in result.items():
        l += v*[k]

    print(round(np.mean(l),2), round(np.std(l),2))
    print(norm_dict(result))

    axs[3].bar(norm_dict(result).keys(), norm_dict(result).values(), 0.5, color='lightblue', hatch='-', label='appreciation')
    axs[3].set_xlabel('Before', fontsize=20)
    axs[0].set_xlabel('Before', fontsize=20)

    # COND
    o_items_p = ["q3:1", "q3:2", "q3:3"]
    o_items_n = ["q4:1", "q4:2", "q4:3"]
    o_items = o_items_p + o_items_n
    a_items_p = ["q3:4", "q4:5", "q4:6"]
    a_items_n = ["q4:4", "q3:5", "q3:6"]
    a_items = a_items_p + a_items_n  
    ## Consistency + Distribution
    # Openness
    df_val[o_items] -= 3
    df_val[o_items_n] = -df_val[o_items_n]
    df_val[o_items] += 3
    # print("\nCronbach alpha: {}".format(pg.cronbach_alpha(data=df_val[o_items])))

    o_counts = []
    for item in o_items:
        o_counts.append(df_val[item].value_counts().to_dict())
    counter = collections.Counter()
    for d in o_counts: 
        counter.update(d)
    result = dict(counter)  

    l = []
    for (k,v) in result.items():
        l += v*[k]

    print(round(np.mean(l),2), round(np.std(l),2))
    print(norm_dict(result))

    axs[1].bar(norm_dict(result).keys(), norm_dict(result).values(), 0.5, color='blue')

    # Appreciation
    df_val[a_items] -= 3 
    df_val[a_items_n] = -df_val[a_items_n]
    df_val[a_items] += 3 
    # print("\nCronbach alpha: {}".format(pg.cronbach_alpha(data=df_val[a_items])))

    a_counts = []
    for item in a_items:
        a_counts.append(df_val[item].value_counts().to_dict())
    counter = collections.Counter()
    for d in a_counts: 
        counter.update(d)
    result = dict(counter)  

    l = []
    for (k,v) in result.items():
        l += v*[k]

    print(round(np.mean(l),2), round(np.std(l),2))
    print(norm_dict(result))

    axs[4].bar(norm_dict(result).keys(), norm_dict(result).values(), 0.5, color='lightblue', hatch='-',)
    axs[4].set_xlabel('During', fontsize=20)
    axs[1].set_xlabel('During', fontsize=20)


    # POST
    o_items_p = ["q5:1", "q5:2"]
    o_items_n = ["q6:1", "q6:2"]
    o_items = o_items_p + o_items_n
    a_items_p = ["q5:3", "q6:4"]
    a_items_n = ["q5:4", "q6:3"]
    a_items = a_items_p + a_items_n   
    ## Consistency + Distribution
    # Openness
    df_val[o_items] -= 3
    df_val[o_items_n] = -df_val[o_items_n]
    df_val[o_items] += 3
    # print("\nCronbach alpha: {}".format(pg.cronbach_alpha(data=df_val[o_items])))

    o_counts = []
    for item in o_items:
        o_counts.append(df_val[item].value_counts().to_dict())
    counter = collections.Counter()
    for d in o_counts: 
        counter.update(d)
    result = dict(counter)  

    l = []
    for (k,v) in result.items():
        l += v*[k]

    print(round(np.mean(l),2), round(np.std(l),2))
    print(norm_dict(result))

    axs[2].bar(norm_dict(result).keys(), norm_dict(result).values(), 0.5, color='blue', label='openness')

    # Appreciation
    df_val[a_items] -= 3 
    df_val[a_items_n] = -df_val[a_items_n]
    df_val[a_items] += 3 
    # print("\nCronbach alpha: {}".format(pg.cronbach_alpha(data=df_val[a_items])))

    a_counts = []
    for item in a_items:
        a_counts.append(df_val[item].value_counts().to_dict())
    counter = collections.Counter()
    for d in a_counts: 
        counter.update(d)
    result = dict(counter)  

    l = []
    for (k,v) in result.items():
        l += v*[k]

    print(round(np.mean(l),2), round(np.std(l),2))
    print(norm_dict(result))

    axs[5].bar(norm_dict(result).keys(), norm_dict(result).values(), 0.5, color='lightblue', hatch='-', label='appreciation')
    axs[5].set_xlabel('After', fontsize=20)
    axs[2].set_xlabel('After', fontsize=20)

    axs[0].set_ylabel('% of responses', fontsize=20)
    # axs[1,0].set_ylabel('% of responses', fontsize=20)
    axs[1].set_title('Openness', fontsize=20)
    axs[4].set_title('Appreciation', fontsize=20)

    axs[0].grid(axis='y')
    axs[1].grid(axis='y')
    axs[2].grid(axis='y')
    axs[3].grid(axis='y')
    axs[4].grid(axis='y')
    axs[5].grid(axis='y')

    plt.show()


    ############ END PLOT 





    # # OVERALL
    # ch_items = ["q7:1", "q7:2", "q7:3", "q7:4" ,"q7:5"]
    # co_items_p = ["q8:1", "q8:2", "q8:3", "q8:4"]
    # co_items_n = ["q8:5", "q8:6", "q8:7", "q8:8"]
    # co_items = co_items_p + co_items_n


    # # Change 
    # print("\nCronbach alpha: {}".format(pg.cronbach_alpha(data=df_val[ch_items])))

    # o_counts = []
    # for item in ch_items:
    #     o_counts.append(df_val[item].value_counts().to_dict())
    # counter = collections.Counter()
    # for d in o_counts: 
    #     counter.update(d)
    # result = dict(counter)  

    # l = []
    # for (k,v) in result.items():
    #     l += v*[k]

    # print(round(np.mean(l),2), round(np.std(l),2))
    # print(norm_dict(result))


    # # Concl
    # df_val[co_items] -= 3 
    # df_val[co_items_n] = -df_val[co_items_n]
    # df_val[co_items] += 3 
    # print("\nCronbach alpha: {}".format(pg.cronbach_alpha(data=df_val[co_items])))

    # a_counts = []
    # for item in co_items:
    #     a_counts.append(df_val[item].value_counts().to_dict())
    # counter = collections.Counter()
    # for d in a_counts: 
    #     counter.update(d)
    # result = dict(counter)  

    # l = []
    # for (k,v) in result.items():
    #     l += v*[k]

    # print(round(np.mean(l),2), round(np.std(l),2))
    # print(norm_dict(result))


    ##### factor analysis

    # # Bartlett’s test 
    # chi_square_value,p_value = calculate_bartlett_sphericity(df_val)
    # print("\nChi^2 value:{}, p-value {}".format(chi_square_value, p_value))

    # # Kaiser-Meyer-Olkin (KMO) Test
    # kmo_all, kmo_model = calculate_kmo(df_val)
    # print("Kaiser-Meyer-Olkin test: {}".format(kmo_model))



    # fa = FactorAnalyzer()
    # fa.fit(df_val)
    # ev, v = fa.get_eigenvalues()
    # print("Eigenvalues")
    # print(ev)

    # # Create scree plot using matplotlib
    # plt.scatter(range(1,df_val.shape[1]+1),ev)
    # plt.plot(range(1,df_val.shape[1]+1),ev)
    # plt.title('Scree Plot')
    # plt.xlabel('Factors')
    # plt.ylabel('Eigenvalue')
    # plt.grid()
    # plt.show()

    # n_fact = len([x for x in ev if x > 1])
    # # n_fact = 2

    # print("\nFactor(s) > 1: {}".format(n_fact))

    # fa = FactorAnalyzer(n_factors=n_fact)
    # fa.set_params(rotation='varimax')
    # fa.fit(df_val)

    # float_formatter = "{:.2f}".format
    # np.set_printoptions(formatter={'float_kind':float_formatter})
    
    # print("\nLoadings")
    # print(fa.loadings_)

    # print()
    # print(fa.get_communalities())

    # tab = [np.append([x], y) for x,y in zip(['SS Loadings', 'Proportion Var', 'Cumulative Var'], fa.get_factor_variance())]
    # print()
    # print(tabulate(tab, headers=[""]+["factor {}".format(x) for x in range(1,n_fact+1)], tablefmt="github"))



    # # mod = """
    # # # measurement model
    # # factor1 =~ y1 + y4 + y6 + y13 + y14 + y16
    # # factor2 =~ y2 + y3 + y9 + y10 + y15
    # # # regressions
    # # factor1 ~ x1
    # # factor2 ~ x1
    # # factor2 ~ factor1
    # # """

    # # model = sem.Model(mod)
    # # model.fit(df, obj="MLW", solver="SLSQP")#, groups=['x1'])
    # # print(model.inspect(mode='list', what="names", std_est=True))

    # # g = sem.semplot(model, "model.png")
    # # print(sem.calc_stats(model))


