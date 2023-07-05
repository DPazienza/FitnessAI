import numpy as np
import pandas as pd


class CalculateAngle:

    AngleDict = {
        'angl_left_elbow': [9, 7, 5],
        'angl_left_shoulder': [7, 5, 11],
        'angl_left_hip': [5, 11, 13],
        'angl_left_knee': [11, 13, 15],
        'angl_right_elbow': [6, 8, 10],
        'angl_right_shoulder': [8, 6, 12],
        'angl_right_hip': [6, 12, 14],
        'angl_right_knee': [12, 14, 16],
    }


    @staticmethod
    def calculate_angle(a, b, c):
        # a,b e c sono array e dovrebbero essere composti da punti x,y, e confidence
        a = np.array(a)  # first
        b = np.array(b)  # mid
        c = np.array(c)  # end

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = int(np.abs(radians * 180.0 / np.pi))

        if angle > 180.0:
            angle = 360 - angle

        return angle

    def calculate_all_angles(self, keypoints):
        angleList = []
        for key in self.AngleDict:
            first_point = keypoints[self.AngleDict[key][0]]
            second_point = keypoints[self.AngleDict[key][1]]
            third_point = keypoints[self.AngleDict[key][2]]
            angle = self.calculate_angle(first_point, second_point, third_point)
            angleList.append(angle)
            confidence = (first_point[2]+second_point[2]+third_point[2])/3
            angleList.append(confidence)
        return angleList
