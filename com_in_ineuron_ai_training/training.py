import os
import tensorflow as tf
from tensorflow.core.protobuf import saver_pb2
from com_in_ineuron_ai_utils.modelArchitecture import nvidia_model
from com_in_ineuron_ai_utils.trainingFileUtils import batch_generator, image_generator

class TrainingModule:

    def __init__(self, modelSavePath, logPath, ckptFileName, epochs, batchSize):
        self.modelSavePath = modelSavePath
        self.logPath = logPath
        self.ckptFileName = ckptFileName
        self.normalisationCont = 0.001
        self.epochs = epochs
        self.batchSize = batchSize

    def trainDrivingModel(self):

        X_train, y_train, X_valid, y_valid, num_train_images, num_val_images = image_generator()
        model = nvidia_model()
        history = model.fit(batch_generator(X_train, y_train, self.batchSize, 1),
                                  steps_per_epoch=num_train_images // self.batchSize,
                                  epochs=self.epochs,
                                  validation_data=batch_generator(X_valid, y_valid, self.batchSize, 0),
                                  validation_steps=num_val_images // self.batchSize,
                                  verbose=1,
                                  shuffle = 1)
        model.save(os.path.join(self.modelSavePath, self.ckptFileName))
        print("Model saved")