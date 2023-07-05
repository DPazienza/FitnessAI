import sys
from collections import deque
import cv2
import numpy as np
import pandas as pd
from keras.saving.saving_api import load_model
from PIL import Image, ImageTk
import tkinter as tk
from tkinter.ttk import Combobox
from Dataset import Dataset
from MoveNet import MoveNet
from angle import CalculateAngle


def saveKeypointsOneFrame(keyPoints):
    listframekeypoints = []
    for point in keyPoints[0][0]:
        listframekeypoints.append(point[0])
        listframekeypoints.append(point[1])
        listframekeypoints.append(point[2])
    listframekeypoints.extend(CalculateAngle().calculate_all_angles(keyPoints[0][0]))
    return listframekeypoints


def calcAccuracyExercise(df, exercise, row):
    angles = np.array(row[51::2])
    conf = np.array(row[52::2])
    confFunc = np.vectorize(lambda v: 0 if v < .2 else 1)
    conf = confFunc(conf)

    df_filtered = df[(df['exercise'] == exercise)]
    mean_errors = df_filtered[df_filtered['type'] == 'errorsMean'].to_numpy()[0][3:]
    # Ordina l'array mean_errors
    sorted_mean_errors = np.sort(mean_errors)

    # Seleziona gli ultimi 4 elementi in mean_errors
    last_4_mean_errors = sorted_mean_errors[-4:]

    # Trova gli indici degli elementi in mean_errors da rimuovere
    indices_to_remove = []
    for error in last_4_mean_errors:
        indices_to_remove.extend(np.where(mean_errors == error)[0])

    # Crea un array con gli stessi elementi di valAngle impostati a 1,5
    relevance = np.ones_like(angles)
    relevance = relevance * 1.25

    # Imposta a 1 gli elementi corrispondenti da rimuovere
    relevance[indices_to_remove] = 0.75
    relevance = relevance * conf

    minAngles = df_filtered[df_filtered['type'] == 'min'].to_numpy()[0][3:]
    maxAngles = df_filtered[df_filtered['type'] == 'max'].to_numpy()[0][3:]

    valPercent = np.array(((angles - minAngles) / maxAngles))
    print(f'valPercent: {valPercent}')
    vfunc = np.vectorize(lambda v: -1 if 0 <= v <= 1 else 0 if (1 < v <= 1.1) or (0.1 <= v < 0) else 1)
    arrow = valPercent.copy()
    arrFunc = np.vectorize(lambda v: '^' if v < .2 else 'v' if v > .8 else '')
    arrow = arrFunc(arrow)
    adjFunc = np.vectorize(
        lambda arrow, min, max, angle: 0 if arrow == '' else max - angle if arrow == 'v' else angle - min)
    adjustment = adjFunc(arrow, minAngles, maxAngles, angles)
    valAcc = vfunc(valPercent)
    valAcc = np.array(valAcc)
    print(f'val angles after elaboration: {valAcc}')
    colors = valAcc.copy()
    colFunc = np.vectorize(lambda v: "green" if v == -1 else "orange" if v == 0 else "red")
    colors = colFunc(colors)
    div = np.count_nonzero(relevance)
    accuracy = np.array(valAcc * relevance).sum() / div if div != 0 else 0
    return accuracy, colors, arrow, adjustment


def textAdjustment(adjustment, angleDict):
    listAdj = []
    listDict = list(angleDict)
    for key in angleDict:
        index = listDict.index(key)
        adj = adjustment[index]
        if adj != 0:
            listAdj.append(f'modificare angolo {key} di {adj} gradi')
    return listAdj


def calculate_distance(array1, array2):
    return np.linalg.norm(array1 - array2)


def checkConfidence(row, minConf):
    conf = np.array(row[52::2])
    confMean = conf.mean()
    return True if confMean > minConf else False


def calculate_position(df, row, exercise):
    angles = np.array(row[51::2])
    df_filtered = df[(df['exercise'] == exercise)]
    start = df_filtered[df_filtered['type'] == 'toStart'].to_numpy()[0][3:]
    middle = df_filtered[df_filtered['type'] == 'toMiddle'].to_numpy()[0][3:]

    distStart = calculate_distance(angles, start)
    distMiddle = calculate_distance(angles, middle)

    distances = [distStart, distMiddle]
    return distances.index(min(distances))


def setFeedback(acc):
    if acc < -.6:
        return "green"
    elif -.6 < acc < .6:
        return "orange"
    else:
        return "red"


class FitensssAi:
    def __init__(self, lightning=False, combo=True):
        self.combo = combo
        self.cap = cv2.VideoCapture(0)
        self.movenet = MoveNet(False)
        self.listKeyPoints = deque(maxlen=15)
        self.dfCentroid = pd.read_csv('datasets/csv/CentroidDataset.csv')
        self.counterToStamp = 0
        self.statusToStamp = 0
        self.exerciseToStamp = 'No exercise recognized'
        self.iStatus = 0
        self.iExercise = 0
        self.dataset = Dataset()
        self.exerciseList = np.array(self.dataset.exerciseList)
        self.window = tk.Tk()
        self.window.geometry("1920x1080")
        self.image_label = tk.Label(self.window)
        self.image_label.pack(side="left")
        self.exercise_label = tk.Label(self.window, text="Exercise:", font=("Arial", 26))
        self.exercise_label.pack()
        self.status_label = tk.Label(self.window, text="Status:", font=("Arial", 26))
        self.status_label.pack()
        self.accuracy_label = tk.Label(self.window, text="Accuracy:", font=("Arial", 26))
        self.accuracy_label.pack()
        self.counter_label = tk.Label(self.window, text="Counter:", font=("Arial", 26))
        self.counter_label.pack()
        self.exercise_combobox = Combobox(self.window, values=list(self.dataset.exerciseList.copy()),
                                          font=("Arial", 26))
        self.exercise_combobox.pack()
        self.canvas = tk.Canvas(self.window, width=200, height=200, bg="white")
        self.canvas.pack()
        # Imposta le dimensioni variabili per il widget Text
        self.text_box = tk.Text(self.window, font=("Arial", 12), height=50, width=10)
        self.text_box.pack()
        if combo:
            self.lstmExerciseRecog = load_model('Models/COMBO LSTM/Combo_LSTM_ExerciseRecog_128_300.h5')
        else:
            self.lstmExerciseRecog = load_model('Models/LSTM/LSTM_ExerciseRecog_128_300.h5')

    def setExercise(self, exercise, exerciseLabel):
        if exercise == exerciseLabel:
            self.iExercise = 50
            self.exerciseToStamp = exerciseLabel
        else:
            self.iExercise -= 1
            if self.iExercise <= 0:
                self.exerciseToStamp = 'No exercise recognized'
                self.iExercise = 0

    def setStatus(self, status):
        changed = False
        if status == self.statusToStamp:
            self.iStatus += 1
            if self.iStatus >= 5:
                self.iStatus = 5

        else:
            self.iStatus -= 1
            if self.iStatus <= 0:
                self.iStatus = 0
                self.statusToStamp = status
                changed = True
        return changed

    def save_and_update_image(self):
        success, frame = self.cap.read()

        if success:
            frame = cv2.flip(frame, 1)
            newframe, keypoints = self.movenet.processFrame(frame, drawing=True)
            if keypoints is not None:
                row = saveKeypointsOneFrame(keypoints)
                self.listKeyPoints.append(row)
                acc = None
                color = "white"
                exerciseVal = ''
                adjustment = []
                selected_exercise = self.exercise_combobox.get()
                self.text_box.delete("1.0", tk.END)

                if len(self.listKeyPoints) >= 15 and checkConfidence(row, 0.2):
                    arrayToPredict = np.array(self.listKeyPoints)
                    array_reshaped = np.reshape(arrayToPredict, (1, 15, 67))

                    if self.combo:
                        exercisePred = self.lstmExerciseRecog.predict([array_reshaped[:, :, 0:51],
                                                                       [array_reshaped[:, :, 51:]]], verbose=0)
                    else:
                        exercisePred = self.lstmExerciseRecog.predict(array_reshaped, verbose=0)

                    exerciseIndex = np.argmax(exercisePred, axis=1)[0]
                    exerciseVal = self.exerciseList[exerciseIndex]
                    self.setExercise(exerciseVal, selected_exercise)
                    if selected_exercise == self.exerciseToStamp:
                        accVal, colors, arrows, adjustment = calcAccuracyExercise(self.dfCentroid, selected_exercise, row)
                        self.movenet.drawing(newframe, keypoints, 0.2, colors, arrows)
                        position = calculate_position(self.dfCentroid, row, selected_exercise)
                        if self.setStatus(position) and self.statusToStamp == 0:
                            self.counterToStamp += 1
                        acc = accVal
                        color = setFeedback(acc)

                print(f'exercise pred: {exerciseVal},exerc counter: {self.iExercise},acc: {acc}, color: {color}')
                listAdj = textAdjustment(adjustment, CalculateAngle.AngleDict)
                for frase in listAdj:
                    self.text_box.insert(tk.END, frase + "\n")
                self.exercise_label.config(text="Exercise: " + self.exerciseToStamp)
                self.status_label.config(text="Status: " + str(self.statusToStamp))
                self.accuracy_label.config(text="Accuracy: " + str(acc))
                self.counter_label.config(text="Counter: " + str(self.counterToStamp))
                self.canvas.configure(bg=color)

            image = Image.fromarray(cv2.cvtColor(newframe, cv2.COLOR_BGR2RGB))
            image = image.resize((1000, 800))

            photo = ImageTk.PhotoImage(image)
            self.image_label.configure(image=photo)
            self.image_label.image = photo

        self.window.after(1, self.save_and_update_image)

    def start(self):

        self.window.after(1000, self.save_and_update_image)
        self.window.mainloop()
        self.cap.release()
        cv2.destroyAllWindows()
        sys.exit(0)


fitnessAI = FitensssAi(combo=True, lightning=False)
fitnessAI.start()
