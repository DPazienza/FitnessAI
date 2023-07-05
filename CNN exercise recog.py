import os
import numpy as np
import pandas as pd
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
from matplotlib import pyplot as plt

exerciseList = ['Bicicletta', 'Crunch', 'Crunch laterali', 'Jumping lunges', 'Leg Raises', 'Mountain climbers',
                'Push up', 'Russian Twist', 'Squat']
df = pd.read_csv('Lightning/FinalDataset.csv')
df = df[df.exercise != 'Burpees']
df = df[df.columns[1:]]

exerciseMap = {label: num for num, label in enumerate(exerciseList)}
lenghtFrame = 40
numFeatures = 67
listDataset = []
listDatasetLabels = []

for exercise in exerciseList:
    counterList = []
    dfExercise = df[df.exercise == exercise]
    for i in range(0, 151):
        dfCounter = dfExercise[dfExercise.counter == i]
        arrayKeyPoints = dfCounter[df.columns[3:]].to_numpy()

        if len(arrayKeyPoints) > 0:
            numframe = len(arrayKeyPoints)
            div = 1
            if numframe < lenghtFrame:
                emptyFrameNum = lenghtFrame - numframe
                arrayKeyPoints = np.concatenate((arrayKeyPoints, np.zeros((emptyFrameNum, numFeatures))), axis=0)
            if numframe > lenghtFrame:
                div = int(numframe / lenghtFrame) + 1
                emptyFrameNum = div * lenghtFrame - numframe
                arrayKeyPoints = np.concatenate((arrayKeyPoints, np.zeros((emptyFrameNum, numFeatures))), axis=0)
            listkeypoints = []
            for keyPoints in arrayKeyPoints[0::div]:
                listkeypoints.append(keyPoints)

            listDataset.append(np.array(listkeypoints))
            listDatasetLabels.append(exerciseMap[exercise])

print('shape listDataset: ({}, {}, {})'.format(len(listDataset), len(listDataset[0]), len(listDataset[0][0])))
print('shape listDatasetLable: ({}, {})'.format(len(listDatasetLabels), len([listDatasetLabels[0]])))

windowsSize = 15
listWindows = []
listWindowsLabels = []
for i in range(0, len(listDataset)):
    for j in range(0, lenghtFrame - windowsSize):
        windows = []
        for keyPoints in listDataset[i][j:j + windowsSize]:
            windows.append(keyPoints)
        listWindows.append(windows)
        listWindowsLabels.append(listDatasetLabels[i])

        # (lenghtFrame- windowsSize ) * 1350 = 25*1500 = 37500
print('shape listWindows: ({}, {}, {})'.format(len(listWindows), len(listWindows[154]), len(listWindows[150][0])))
print('shape listWindowsLabels: ({}, {})'.format(len(listWindowsLabels), len([listWindowsLabels[0]])))

Xtrain, Xtest, Ytrain, Ytest = train_test_split(listWindows, listWindowsLabels, test_size=0.15, shuffle=True)  # da provare shaffle false
Ytrain = to_categorical(Ytrain, num_classes=len(exerciseList)).astype(int)
Ytest = to_categorical(Ytest, num_classes=len(exerciseList)).astype(int)

Xtrain = np.array(Xtrain)
Xtest = np.array(Xtest)

#Rete convoluzionale
import tensorflow as tf


epochs = 300
batchSize = 128
model_name = 'CNN_ExerciseRecog_{}_{}.h5'.format(batchSize, epochs)
logDir = os.path.join('Logs', 'CNN', model_name)
model_path = os.path.join('Models', 'CNN', model_name)
if os.path.exists(model_path):
    model = load_model(model_path)
else:
    # Definizione dell'architettura della CNN
    tb_callback = TensorBoard(log_dir=logDir)
    model = Sequential()
    model.add(Conv1D(32, 3, activation='relu', input_shape=(windowsSize, numFeatures)))  # Layer Conv1D con 32 filtri e kernel size 3
    model.add(MaxPooling1D(2))  # Layer di MaxPooling1D con pool size 2
    model.add(Flatten())
    model.add(Dense(len(exerciseList), activation='sigmoid'))
    # Compilazione del modello
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Addestramento del modello con i tuoi dati
    model.fit(Xtrain, Ytrain, epochs=epochs, batch_size=batchSize)
    model.save(model_path)
    print('Model saved:', model_path)

# Valutazione del modello sui dati di test

test_loss, test_acc = model.evaluate(Xtest, Ytest)
YtestPredict = model.predict(Xtest)
exerciseList = np.array(exerciseList)
accuracy = accuracy_score(exerciseList[np.argmax(Ytest, axis=1).tolist()],
                          exerciseList[np.argmax(YtestPredict, axis=1).tolist()])

precision = precision_score(exerciseList[np.argmax(Ytest, axis=1).tolist()],
                            exerciseList[np.argmax(YtestPredict, axis=1).tolist()],
                            average='macro', zero_division=0)

recall = recall_score(exerciseList[np.argmax(Ytest, axis=1).tolist()],
                      exerciseList[np.argmax(YtestPredict, axis=1).tolist()],
                      average='macro')

f1 = f1_score(exerciseList[np.argmax(Ytest, axis=1).tolist()],
              exerciseList[np.argmax(YtestPredict, axis=1).tolist()],
              average='macro')

ConfusionMatrixDisplay.from_predictions(exerciseList[np.argmax(Ytest, axis=1).tolist()],
                                        exerciseList[np.argmax(YtestPredict, axis=1).tolist()],
                                        cmap=plt.cm.Blues)

print(f'epoch: {epochs}, batchsize = {batchSize}')
print(f'test_acc: {test_acc}')
print(f'precision: {precision}')
print(f'recall: {recall}')
print(f'f1: {f1}')
