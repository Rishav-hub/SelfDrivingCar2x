import tensorflow as tf
import scipy.misc
from com_in_ineuron_ai_utils import modelArchitecture
import cv2
import math
import numpy as np
from PIL import Image
from com_in_ineuron_ai_utils.trainingFileUtils import img_preprocess

# from tensorflow.keras import load_model


class PredcitionOnDataset:

    def __init__(self, modelPath, steeringImagePath, textDataFilePath, datasetFolderPath):
        self.modelPath = modelPath
        self.steeringImagePath = steeringImagePath
        self.txtDataFilePath = textDataFilePath
        self.datasetFolderPath = datasetFolderPath
        self.showRoadLayoutFrameName = "Road Layout"
        self.showSteeringWheelFrameName = "Car Steering"

    def predictorFunc(self):

        cv2.namedWindow(self.showRoadLayoutFrameName, cv2.WINDOW_NORMAL)
        img = cv2.imread(self.steeringImagePath, 0)
        rows, cols = img.shape

        smoothed_angle = 0

        actData = []
        predData = []
        with open(self.txtDataFilePath) as f:
            for line in f:
                actData.append(self.datasetFolderPath + line.split()[0])
                predData.append(float(line.split()[1]) * scipy.pi / 180)

        # Total number of images
        num_images = len(actData)

        i = math.ceil(num_images * 0.8)
        print("Prediction Starting From this frame:" + str(i))

        while cv2.waitKey(10) != ord('q'):
            # inpFrame = scipy.misc.imread(self.datasetFolderPath + str(i) + ".jpg", mode="RGB")
            inpFrame = cv2.imread(self.datasetFolderPath + str(i) + ".jpg")
            # image = scipy.misc.imresize(inpFrame[-150:], [66, 200]) / 255.0
            image = np.asarray(inpFrame)
            image = img_preprocess(image)
            image = np.array([image])
            model = tf.keras.models.load_model(self.modelPath)
            degrees = float(model(image)) * 180 / scipy.pi
            # degrees = model.predict(image[None, :, :, :])[0][0] * 180 / scipy.pi
            # degrees = modelArchitecture.y.eval(feed_dict={modelArchitecture.x: [image], modelArchitecture.keep_prob: 1.0})[0][0] * 180.0 / scipy.pi
            print("Moving angle of steering: " + str(degrees) + " (predicted)\t" + str(predData[i] * 180 / scipy.pi) + " (original)")
            cv2.resizeWindow(self.showRoadLayoutFrameName, 1000, 1000)
            cv2.moveWindow(self.showRoadLayoutFrameName, 40, 30)
            cv2.imshow(self.showRoadLayoutFrameName, cv2.cvtColor(inpFrame, cv2.COLOR_RGB2BGR))

            smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(
                degrees - smoothed_angle)
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -smoothed_angle, 1)
            steeringWheel = cv2.warpAffine(img, M, (cols, rows))
            cv2.moveWindow(self.showSteeringWheelFrameName, 10, 20)
            cv2.imshow(self.showSteeringWheelFrameName, steeringWheel)
            i += 1

        cv2.destroyAllWindows()
