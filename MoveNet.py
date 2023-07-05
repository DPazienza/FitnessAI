import os
import sys
import keyboard
import tensorflow as tf
import numpy as np
import cv2
import mediapipe as mp
from keras.models import load_model

from angle import CalculateAngle

EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}


class MoveNet:
    def __init__(self, lightning):
        self.lightning = lightning
        self.moveNet = None
        self.load_model_movenet()

    def load_model_movenet(self):
        if self.lightning:
            model_path = 'movenet model/lite-model_movenet_singlepose_lightning_3.tflite'
        else:
            model_path = 'movenet model/lite-model_movenet_singlepose_thunder_3.tflite'

        try:
            self.moveNet = tf.lite.Interpreter(model_path=model_path)
            self.moveNet.allocate_tensors()
            return True
        except (FileNotFoundError, ValueError) as e:
            print(f"Failed to load MoveNet model: {str(e)}")
            return False

    @staticmethod
    def drawKeyPoints(frame, keyPoints, confidence):
        y, x, c = frame.shape
        shaped = np.squeeze(np.multiply(keyPoints, [y, x, 1]))

        for kp in shaped:
            ky, kx, kconfidence = kp
            if kconfidence > confidence:
                cv2.circle(frame, (round(kx), round(ky)), 4, (0, 255, 0), -1)

    @staticmethod
    def drawConnector(frame, keyPoints, edges, confidence):
        y, x, c = frame.shape
        shaped = np.squeeze(np.multiply(keyPoints, [y, x, 1]))

        for edge, color in edges.items():
            p1, p2 = edge
            y1, x1, c1 = shaped[p1]
            y2, x2, c2 = shaped[p2]
            c = min(c1, c2)

            if c > confidence:
                cv2.line(frame, (round(x1), round(y1)), (round(x2), round(y2)), (255, 0, 0), 1)

    @staticmethod
    def draw_angles(frame, keypoints, minConf, colors=None, arrows=None):
        y, x, c = frame.shape
        shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))
        angle_dict = CalculateAngle.AngleDict
        print(f'colors: {colors}')
        print(f'colors: {arrows}')
        listDict = list(angle_dict)
        for key in angle_dict:
            first_point = shaped[angle_dict[key][0]]
            second_point = shaped[angle_dict[key][1]]
            third_point = shaped[angle_dict[key][2]]
            if colors is not None and arrows is not None:
                index = listDict.index(key)
                color = colors[index]
                arrow = arrows[index]
                rgbColor = (0, 165, 255) if color == "orange" else (255, 0, 0) if color == "red" else (0, 255, 0)

                # Calcola l'angolo utilizzando il metodo calculate_angle della classe CalculateAngle
                angle_calculator = CalculateAngle()
                angle = round(angle_calculator.calculate_angle(first_point, second_point, third_point))
                confidence = min(first_point[2], second_point[2], third_point[2])

                if confidence >= minConf:
                    # Calcola il centro del semicerchio
                    center = (round(second_point[1]), round(second_point[0]))
                    # Disegna l'angolo sul frame
                    cv2.putText(frame, f'{angle} {arrow}', (round(center[0]), round(center[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                rgbColor, 2)
        return frame

    def drawing(self, frame, keypoints, minConf, colors=None, arrows=None):
        self.drawKeyPoints(frame, keypoints, minConf)
        self.drawConnector(frame, keypoints, EDGES, minConf)
        self.draw_angles(frame, keypoints[0][0], minConf, colors, arrows)

    def processFrame(self, frame, drawing=False, angleParams=None):
        if self.moveNet is None:
            return frame, None

        img = frame.copy()
        if self.lightning:
            img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192, 192)
        else:
            img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 256, 256)

        inputImage = tf.cast(img, dtype=tf.float32)
        inputDetails = self.moveNet.get_input_details()
        outputDetails = self.moveNet.get_output_details()
        self.moveNet.set_tensor(inputDetails[0]['index'], np.array(inputImage))
        self.moveNet.invoke()
        keyPointsWithScores = self.moveNet.get_tensor(outputDetails[0]['index'])
        if drawing:
            self.drawing(frame, keyPointsWithScores, 0.3, angleParams)
        return frame, keyPointsWithScores

    def useVideoFile(self, path, funcDataAugmentation=None):
        if not os.path.exists(path):
            print("File not found.")
            return 0, []

        cap = cv2.VideoCapture(path)
        listKeyPoints = []

        if not self.load_model_movenet():
            return listKeyPoints

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            if funcDataAugmentation is not None:
                frame = funcDataAugmentation(frame)
            newFrame, keyPoints = self.processFrame(frame, drawing=False)
            listKeyPoints.append(keyPoints)

        cap.release()
        return listKeyPoints

    def useCam(self):
        if not self.load_model_movenet():
            return

        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame, keypoints = self.processFrame(frame)
            cv2.imshow('Movenet Lightning', frame)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

            try:
                if keyboard.is_pressed('q'):
                    print('You exited!')
                    break
            except:
                break

        cap.release()
        cv2.destroyAllWindows()
        sys.exit(0)
