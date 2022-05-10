#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.api import stats

INFILE_1 = "../data/listenbrainz/results/logs_analysis.csv"
INFILE_2 = "../data/listenbrainz/results/logs_diff_analysis.csv"

COLS = ["username", "group", "phase"]

COLS_VAR = ["EM_genres_unique_p", "EM_genres_count_p",
            "EM_artists_unique_p", "EM_artists_count_p",
            "EM_tracks_unique_p", "EM_tracks_count_p"]

if __name__ == "__main__":

    df = pd.read_csv(INFILE_1)

    df = df[COLS + COLS_VAR]

    df_merge = df[df.phase == 'PRE'].merge(df[df.phase == 'POST'], on=('username', 'group'))
    formula = "EM_genres_unique_p_y ~ C(group) + EM_genres_unique_p_x"
    model = ols(formula, data=df_merge).fit()
    print(model.summary())
