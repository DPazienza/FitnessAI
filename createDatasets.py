import functools
import random
import cv2
import numpy as np
import pandas as pd
import os
from MoveNet import MoveNet
from angle import CalculateAngle

COLUMNS_ONE_FRAME = ['exercise', 'counter', 'num-frame',
                     'nose x', 'nose y', 'nose c',
                     'sx eye x', 'sx eye y', 'sx eye c',
                     'dx eye x', 'dx eye y', 'dx eye c',
                     'sx ear x', 'sx ear y', 'sx ear c',
                     'dx ear x', 'dx ear y', 'dx ear c',
                     'sx shoulder x', 'sx shoulder y', 'sx shoulder c',
                     'dx shoulder x', 'dx shoulder y', 'dx shoulder c',
                     'sx elbow x', 'sx elbow y', 'sx elbow c',
                     'dx elbow x', 'dx elbow y', 'dx elbow c',
                     'sx wrist x', 'sx wrist y', 'sx wrist c',
                     'dx wrist x', 'dx wrist y', 'dx wrist c',
                     'sx hip x', 'sx hip y', 'sx hip c',
                     'dx hip x', 'dx hip y', 'dx hip c',
                     'sx knee x', 'sx knee y', 'sx knee c',
                     'dx knee x', 'dx knee y', 'dx knee c',
                     'sx ankle x', 'sx ankle y', 'sx ankle c',
                     'dx ankle x', 'dx ankle y', 'dx ankle c',
                     'angl_left_elbow', 'angl_left_elbow conf', 'angl_left_shoulder', 'angl_left_shoulder conf',
                     'angl_left_hip', 'angl_left_hip conf', 'angl_left_knee', 'angl_left_knee conf',
                     'angl_right_elbow', 'angl_right_elbow conf', 'angl_right_shoulder',
                     'angl_right_shoulder conf',
                     'angl_right_hip', 'angl_right_hip conf', 'angl_right_knee', 'angl_right_knee conf']


def saveKeypointsOneFrame(counter, keyPoints, dataFrame, exercise_, save=True):
    frameNum = len(keyPoints)
    finalList = []
    for i in range(0, frameNum - 1):
        listframekeypoints = []
        if save:
            listframekeypoints.append(exercise_)
            listframekeypoints.append(counter)
            listframekeypoints.append(i)
        for point in keyPoints[i][0][0]:
            listframekeypoints.append(point[0])
            listframekeypoints.append(point[1])
            listframekeypoints.append(point[2])
        listframekeypoints.extend(CalculateAngle().calculate_all_angles(keyPoints[i][0][0]))
        if save:
            dataFrame.loc[len(dataFrame)] = listframekeypoints
        else:
            finalList.append(listframekeypoints)
    return finalList


def flipVideo(frame):
    return cv2.flip(frame, 1)


def generate_interpolated_frame(frame1, frame2):
    # Applica l'algoritmo di interpolazione dei frame (frame blending)
    alpha = 0.5  # Fattore di miscelazione
    interpolated_frame = cv2.addWeighted(frame1, alpha, frame2, alpha, 0.0)
    interpolated_frame = interpolated_frame.astype(np.uint8)
    return interpolated_frame


def modifica_velocita_video(input_file, output_file, speed_factor):
    # Apri il video di input
    video_capture = cv2.VideoCapture(input_file)

    # Ottieni le dimensioni del video
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)

    # Calcola il nuovo frame rate
    new_fps = fps * speed_factor

    # Crea il writer per il video di output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, new_fps, (frame_width, frame_height))

    # Leggi i frame del video di input e scrivi i frame modificati nel video di output
    frame_buffer = []

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Aggiungi il frame al buffer
        frame_buffer.append(frame)

        # Rallenta il video generando frame intermedi
        if len(frame_buffer) == 2:
            # Scrivi il frame corrente nel video di output
            out.write(frame_buffer[0])

            # Rallenta il video generando frame intermedi
            for _ in range(int(1 / speed_factor) - 1):
                interpolated_frame = generate_interpolated_frame(frame_buffer[0], frame_buffer[1])
                # Scrivi i frame intermedi nel video di output
                out.write(interpolated_frame)

            frame_buffer.pop(0)  # Rimuovi il frame più vecchio dal buffer

    # Scrivi l'ultimo frame nel video di output
    if frame_buffer:
        out.write(frame_buffer[0])

    # Rilascia le risorse
    video_capture.release()
    out.release()



def custom_augmentation(frame, angle):
    # Determina l'angolo di rotazione casuale tra -10 e 10 gradi
    # angle = random.uniform(-10, 10)

    # Ottieni le dimensioni del frame
    height, width = frame.shape[:2]

    # Calcola il centro del frame
    center = (width // 2, height // 2)

    # Definisci la matrice di trasformazione per la rotazione
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Applica la trasformazione di rotazione al frame
    rotated_frame = cv2.warpAffine(frame, M, (width, height))

    return rotated_frame


def custom_augmentation_rescale(frame, scale_factor):
    # Determina il fattore di ridimensionamento casuale tra 0.8 e 1.2
    # scale_factor = random.uniform(0.8, 1.2)

    # Ottieni le dimensioni del frame
    height, width = frame.shape[:2]

    # Calcola le nuove dimensioni dopo il ridimensionamento
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)

    # Ridimensiona il frame utilizzando l'interpolazione bilineare
    resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # Ritaglia il frame alla dimensione originale
    top = (new_height - height) // 2
    left = (new_width - width) // 2
    cropped_frame = resized_frame[top:top + height, left:left + width]

    return cropped_frame


def processDataAugmentation(exercise, video_file_path, dataframeExercise, counter, moveNet):
    listKeyPoints = moveNet.useVideoFile(video_file_path)
    saveKeypointsOneFrame(counter, listKeyPoints, dataframeExercise, exercise)
    print(f"Exercise: {exercise}, Video: {counter} - Processing completed")

    listKeyPoints = moveNet.useVideoFile(video_file_path, flipVideo)
    saveKeypointsOneFrame(counter + 1, listKeyPoints, dataframeExercise, exercise)
    print(f"Exercise: {exercise}, Video: {counter + 1} - Processing completed")

    rotateFunc = functools.partial(custom_augmentation, angle=random.uniform(-10, 10))
    listKeyPoints = moveNet.useVideoFile(video_file_path, rotateFunc)
    saveKeypointsOneFrame(counter + 2, listKeyPoints, dataframeExercise, exercise)
    print(f"Exercise: {exercise}, Video: {counter + 2} - Processing completed")

    scaleFunc = functools.partial(custom_augmentation_rescale, scale_factor=random.uniform(0.8, 1.2))
    listKeyPoints = moveNet.useVideoFile(video_file_path, scaleFunc)
    saveKeypointsOneFrame(counter + 3, listKeyPoints, dataframeExercise, exercise)
    print(f"Exercise: {exercise}, Video: {counter + 3} - Processing completed")


def process_video(exercise, video_path, dataframeExercise, moveNet):
    video_files = os.listdir(os.path.join(video_path, exercise))

    for video_file in video_files:
        counter = video_files.index(video_file)
        video_file_path = os.path.join(video_path, exercise, video_file)

        processDataAugmentation(exercise, video_file_path, dataframeExercise, counter*12, moveNet)
        speed = 'Workout/tmp/speed.mp4'  # File di output
        slow = 'Workout/tmp/slow.mp4'  # File di output

        # Creazione della directory se non esiste
        output_dir = os.path.dirname(speed)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            # Creazione della directory se non esiste
        output_dir = os.path.dirname(slow)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        modifica_velocita_video(video_file_path, slow, 0.5)
        processDataAugmentation(exercise, slow, dataframeExercise, (counter*12)+4, moveNet)

        modifica_velocita_video(video_file_path, speed, 1.5)
        processDataAugmentation(exercise, speed, dataframeExercise, (counter*12)+8, moveNet)


if __name__ == '__main__':
    video_path = 'Workout/Video elaborati'
    csv_path = 'datasets/csv'
    moveNet = MoveNet(False)

    list_exercise = os.listdir(video_path)
    print(list_exercise)
    dataframe_keypoints = pd.DataFrame(columns=COLUMNS_ONE_FRAME)

    for exercise in list_exercise:
        process_video(exercise, video_path, dataframe_keypoints, moveNet)
        print(dataframe_keypoints)

    dataframe_keypoints.to_csv(csv_path + '/FinalDataset.csv')

    print("Keypoints")
    print(dataframe_keypoints)
