import os
import sys
from io import StringIO

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from keras.utils import to_categorical
from keras.utils.layer_utils import print_summary
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.utils import resample


def calculateWindows(listDataset, listDatasetLabels, windowsSize):
    listWindows = []
    listWindowsLabels = []
    for i in range(0, len(listDataset)):
        for j in range(0, len(listDataset[i]) - windowsSize):
            windows = []
            for keyPoints in listDataset[i][j:j + windowsSize]:
                windows.append(keyPoints)
            listWindows.append(windows)
            listWindowsLabels.append(listDatasetLabels[i])
    return listWindows, listWindowsLabels


class Dataset:
    def __init__(self):

        self.df = pd.read_csv('datasets/csv/FinalDataset.csv')
        self.df = self.df[self.df.columns[1:]]
        self.numFeatures = len(self.df.to_numpy()[0]) - 3
        self.exerciseList = ['Bicicletta', 'Burpees', 'Crunch', 'Crunch laterali', 'Jumping lunges', 'Leg Raises',
                             'Mountain climbers', 'Push up', 'Russian Twist', 'Squat']
        self.exerciseMap = {label: num for num, label in enumerate(self.exerciseList)}

    def calculateListDataset(self, lenghtFrame=15):

        listDataset = []
        listDatasetLabels = []

        for exercise in self.exerciseList:
            dfExercise = self.df[self.df.exercise == exercise]
            max_counter = dfExercise['counter'].max()
            print(max_counter)
            for i in range(0, max_counter):
                dfCounter = dfExercise[dfExercise.counter == i]
                arrayKeyPoints = dfCounter[self.df.columns[3:]].to_numpy()

                if len(arrayKeyPoints) > 0:
                    numframe = len(arrayKeyPoints)
                    if numframe > lenghtFrame:
                        # emptyFrameNum = lenghtFrame - numframe
                        # arrayKeyPoints = np.concatenate((arrayKeyPoints, np.zeros((emptyFrameNum, self.numFeatures))),
                        #                                 axis=0)
                        listkeypoints = []
                        for keyPoints in arrayKeyPoints:
                            listkeypoints.append(keyPoints)

                        listDataset.append(np.array(listkeypoints))
                        listDatasetLabels.append(self.exerciseMap[exercise])

        print('shape listDataset: ({}, {})'.format(len(listDataset), np.array(listDataset[0]).shape))
        print('shape listDatasetLabel: ({}, {})'.format(len(listDatasetLabels), len([listDatasetLabels[0]])))
        return listDataset, listDatasetLabels

    def calculateStatus(self, exercise, columns):
        dfExercise = self.df[self.df['exercise'] == exercise]
        max_counter = dfExercise['counter'].max()

        dfExercise['status'] = np.nan  # Creiamo una colonna 'status' inizialmente con valori NaN

        for i in range(max_counter):
            dfCounter = dfExercise[dfExercise['counter'] == i]
            arrayKeyPoints = dfCounter.iloc[:, 3:].to_numpy()
            numFrames = len(arrayKeyPoints)

            if numFrames > 0:
                val = np.round(((np.arange(1, numFrames + 1) / numFrames) * 100) / 12)
                status = np.where((val == 0) | (val == 1) | (val == 2) | (val == 3) | (val == 5) | (val == 6) | (val == 7) | (val == 8), 0, 1)
                dfExercise.loc[dfCounter.index, 'status'] = status

        dfExercise = dfExercise[columns]
        print(f"calculate status of {exercise} completed")
        return dfExercise

    def saveMetrics(self, path, accuracy, precision, recall, f1, matrix, model):
        metric_image = Image.new("RGB", (500, 300), (255, 255, 255))
        draw = ImageDraw.Draw(metric_image)

        accuracy = round(accuracy, 3)
        precision = round(precision, 3)
        recall = round(recall, 3)
        f1 = round(f1, 3)

        font = ImageFont.truetype("arialbd.ttf", 16)

        # Disegna la tabella delle metriche
        draw.text((50, 50), "Metriche", font=font, fill=(0, 0, 0))
        draw.text((100, 100), f"\u2022 Accuracy: {accuracy:.3f}", font=font, fill=(0, 0, 0))
        draw.text((100, 130), f"\u2022 Precision: {precision:.3f}", font=font, fill=(0, 0, 0))
        draw.text((100, 160), f"\u2022 Recall: {recall:.3f}", font=font, fill=(0, 0, 0))
        draw.text((100, 190), f"\u2022 F1-Score: {f1:.3f}", font=font, fill=(0, 0, 0))

        # Salvataggio della prima foto
        metric_image.save(os.path.join(path, "metric_image.jpg"))

        image_width = 900
        image_height = 600
        background_color = (255, 255, 255)
        # Crea un oggetto StringIO per catturare l'output del summary
        summary_buffer = StringIO()
        if model is not None:
            # Salva l'output del summary nel buffer
            print_summary(model, print_fn=lambda x: summary_buffer.write(x + '\n'))

            # Ottieni il summary come una stringa di testo
            summary_text = summary_buffer.getvalue()

            # Chiudi il buffer
            summary_buffer.close()

            # Crea l'immagine
            image = Image.new("RGB", (image_width, image_height), background_color)

            # Crea un oggetto ImageDraw per disegnare sulla nuova immagine
            draw = ImageDraw.Draw(image)

            # Imposta il font e le dimensioni del testo
            font = ImageFont.truetype("arial.ttf", 16)

            # Disegna il testo sulla nuova immagine
            draw.text((50, 50), summary_text, font=font, fill=(0, 0, 0))

            # Salva l'immagine su disco
            image.save(os.path.join(path, "model_description.png"))


         # Crea l'oggetto ConfusionMatrixDisplay
        cm_display = ConfusionMatrixDisplay(matrix, display_labels=range(0, 10))

        # Genera il grafico della matrice di confusione
        cm_display.plot(cmap=plt.cm.Blues)

        # Aggiungi il titolo al grafico
        plt.title('Matrice di Confusione')

        plt.tight_layout()

        # Salvataggio della seconda foto
        plt.savefig(os.path.join(path, 'confusion_matrix.jpg'))
        plt.close()

    def makeTrainTestForDL(self, windowsSize):
        pathNumpy = 'datasets/npy'
        if len(os.listdir(pathNumpy)) < 1:
            listDataset, listDatasetLabels = self.calculateListDataset()
            listWindows, listWindowsLabels = calculateWindows(listDataset=listDataset,
                                                              listDatasetLabels=listDatasetLabels,
                                                              windowsSize=windowsSize)

            Xtrain, Xtest, Ytrain, Ytest = train_test_split(listWindows,
                                                            listWindowsLabels, test_size=0.3,
                                                            shuffle=True)

            Xtrain = np.array(Xtrain)
            Ytrain = np.array(Ytrain)

            unique_labels, counts = np.unique(Ytrain, return_counts=True)
            print(f'Number of occurrences before balancing')
            # Stampa il conteggio delle occorrenze per ogni etichetta
            for label, count in zip(unique_labels, counts):
                print(f'{label}: {count} occorrenze')

            # Bilanciamento delle classi tramite undersampling
            min_samples = min(np.bincount(Ytrain))
            Xtrain_balanced = []
            Ytrain_balanced = []
            for label in np.unique(Ytrain):
                indices = np.where(Ytrain == label)[0]
                indices_balanced = resample(indices, replace=False, n_samples=min_samples, random_state=42)
                Xtrain_balanced.append(Xtrain[indices_balanced])
                Ytrain_balanced.append(Ytrain[indices_balanced])

            Xtrain_balanced = np.concatenate(Xtrain_balanced)
            Ytrain_balanced = np.concatenate(Ytrain_balanced)

            unique_labels, counts = np.unique(Ytrain_balanced, return_counts=True)

            # Stampa il conteggio delle occorrenze per ogni etichetta
            print(f'Number of occurrences after balancing')
            for label, count in zip(unique_labels, counts):
                print(f'{label}: {count} occorrenze')

            Ytrain_balanced = to_categorical(Ytrain_balanced, num_classes=len(self.exerciseList)).astype(int)
            Ytest = to_categorical(Ytest, num_classes=len(self.exerciseList)).astype(int)

            Xtrain_balanced = np.array(Xtrain_balanced)
            Xtest = np.array(Xtest)
            np.save(os.path.join(pathNumpy, 'XTrain.npy'), Xtrain_balanced)
            np.save(os.path.join(pathNumpy, 'XTest.npy'), Xtest)
            np.save(os.path.join(pathNumpy, 'YTrain.npy'), Ytrain_balanced)
            np.save(os.path.join(pathNumpy, 'Ytest.npy'), Ytest)

        else:
            Xtrain_balanced = np.load(os.path.join(pathNumpy, 'XTrain.npy'))
            Xtest = np.load(os.path.join(pathNumpy, 'XTest.npy'))
            Ytrain_balanced = np.load(os.path.join(pathNumpy, 'YTrain.npy'))
            Ytest = np.load(os.path.join(pathNumpy, 'Ytest.npy'))
        print(f'len of Xtrain_balanced: {len(Xtrain_balanced)}')
        return Xtrain_balanced, Xtest, Ytrain_balanced, Ytest

