import os
import numpy as np
from Dataset import Dataset
import pandas as pd


def calcola_valori_estremi_con_errore(df, angolo_col, conf_col, conf):
    # Filtra il dataframe per l'esercizio e la confidenza > 0.3 per l'angolo specificato
    df_filtrato = df[(df[conf_col] > conf)]

    # Calcola il primo quartile (25%) e il terzo quartile (75%)
    q1 = np.percentile(df_filtrato[angolo_col], 25)
    q3 = np.percentile(df_filtrato[angolo_col], 75)

    # Calcola l'IQR
    iqr = q3 - q1

    # Calcola i limiti inferiore e superiore per identificare gli outlier
    limite_inferiore = q1 - 1.5 * iqr
    limite_superiore = q3 + 1.5 * iqr

    # Filtra il dataframe per rimuovere gli outlier
    df_filtrato = df_filtrato[
        (df_filtrato[angolo_col] >= limite_inferiore) & (df_filtrato[angolo_col] <= limite_superiore)]

    # Calcola il valore massimo e minimo nel dataframe filtrato
    valore_massimo = df_filtrato[angolo_col].max()
    valore_minimo = df_filtrato[angolo_col].min()

    mean = int(df_filtrato[angolo_col].mean())
    errori = df_filtrato[angolo_col] - mean
    squared_errors = errori ** 2
    errMean = np.sqrt(squared_errors.mean())

    return valore_massimo, valore_minimo, errMean


anglecsvPath = 'datasets/csv/AngleDataset.csv'
centroidcsvPath = 'datasets/csv/CentroidDataset.csv'
COLUMNS_ONE_FRAME = ['exercise', 'counter', 'status',
                     'angl_left_elbow', 'angl_left_elbow conf', 'angl_left_shoulder', 'angl_left_shoulder conf',
                     'angl_left_hip', 'angl_left_hip conf', 'angl_left_knee', 'angl_left_knee conf',
                     'angl_right_elbow', 'angl_right_elbow conf', 'angl_right_shoulder', 'angl_right_shoulder conf',
                     'angl_right_hip', 'angl_right_hip conf', 'angl_right_knee', 'angl_right_knee conf']
COLUMNS_CENTROID = ['exercise', 'type', 'angl_left_elbow', 'angl_left_shoulder', 'angl_left_hip',
                    'angl_left_knee', 'angl_right_elbow', 'angl_right_shoulder', 'angl_right_hip', 'angl_right_knee']
data = Dataset()

if os.path.exists(anglecsvPath):
    df = pd.read_csv(anglecsvPath)
else:
    df = pd.DataFrame(columns=COLUMNS_ONE_FRAME)

    for exercise in data.exerciseList:
        dfExercise = data.calculateStatus(exercise, columns=COLUMNS_ONE_FRAME)
        df = pd.concat([df, dfExercise], ignore_index=True)

    df.to_csv(anglecsvPath)

dfCentroid = pd.DataFrame(columns=COLUMNS_CENTROID)
for exercise in data.exerciseList:
    exerciseDf = df[df.exercise == exercise]
    noPercDf = exerciseDf
    centroid0 = []
    centroid1 = []
    for status in (0, 1):
        percDf = exerciseDf[exerciseDf.status == status]
        percDf = percDf.drop(percDf.columns[0:3], axis=1)
        for i in (1, 3, 5, 7, 9, 11, 13, 15):
            minMaxdf = percDf[percDf[percDf.columns[i]] >= .3]
            mean = int(minMaxdf[percDf.columns[i - 1]].mean())
            if status == 0:
                centroid0.append(round(mean))
            if status == 1:
                centroid1.append(round(mean))

    noPercDf = noPercDf.drop(noPercDf.columns[0:3], axis=1)
    errorsMean = []
    maxArray = []
    minArray = []
    for i in (1, 3, 5, 7, 9, 11, 13, 15):
        minMaxdf = noPercDf[noPercDf[noPercDf.columns[i]] >= .4]
        maxVal, minVal, errMean = calcola_valori_estremi_con_errore(minMaxdf, noPercDf.columns[i - 1], noPercDf.columns[i], .4)
        errorsMean.append(round(errMean))
        maxArray.append(round(maxVal))
        minArray.append(round(minVal))

    dfCentroid.loc[len(dfCentroid)] = np.concatenate(([exercise], ['toStart'], centroid0))
    dfCentroid.loc[len(dfCentroid)] = np.concatenate(([exercise], ['toMiddle'], centroid1))
    dfCentroid.loc[len(dfCentroid)] = np.concatenate(([exercise], ['errorsMean'], errorsMean))
    dfCentroid.loc[len(dfCentroid)] = np.concatenate(([exercise], ['min'], minArray))
    dfCentroid.loc[len(dfCentroid)] = np.concatenate(([exercise], ['max'], maxArray))

dfCentroid.to_csv(centroidcsvPath)
