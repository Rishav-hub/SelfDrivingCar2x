from com_in_ineuron_ai_prediction.prediction import PredcitionOnDataset
from com_in_ineuron_ai_training.training import TrainingModule


def trainModel():
    # ============================================= Training Params ====================================================== #

    modelSavePath = './models'
    logPath = './trainingLogs'
    ckptFileName = "model.h5"
    epochs = 5
    batchSize = 100

    # Initialise the basic params
    tranngModleObj = TrainingModule(modelSavePath, logPath, ckptFileName, epochs, batchSize)

    # Start training the model
    tranngModleObj.trainDrivingModel()


def makePrediction():
    # ============================================= Prediction Params ====================================================== #

    userChoice = input("Select 1 for smallerDataset and 2 for LargeDataset")
    while userChoice.isdigit():
        userChoice = int(userChoice)
        if userChoice in [1,2]:
            if userChoice == 1:
                modelPath = "models/model.h5"
                steeringImagePath = './dataSets/drivingWheel.jpg'
                textDataFilePath = "./dataSets/TrainTestDataSmall2_2/data.txt"
                datasetFolderPath = "./dataSets/TrainTestDataSmall2_2/"
                predctnOnDtst = PredcitionOnDataset(modelPath, steeringImagePath, textDataFilePath, datasetFolderPath)
                predctnOnDtst.predictorFunc()
                break
            elif userChoice == 2:
                modelPath = "models/model.h5"
                steeringImagePath = './dataSets/drivingWheel.jpg'
                textDataFilePath = "./dataSets/TrainTestDataLarge3_9/data.txt"
                datasetFolderPath = "./dataSets/TrainTestDataLarge3_9/"
                predctnOnDtst = PredcitionOnDataset(modelPath, steeringImagePath, textDataFilePath, datasetFolderPath)
                predctnOnDtst.predictorFunc()
                break
        else:
            continue


if __name__ == "__main__":
    userChoice = input("Select \n1 for Training \nand \n2 for Prediction")
    if userChoice.isdigit():
        userChoice = int(userChoice)
        if userChoice == 1:
            trainModel()
        elif userChoice == 2:
            makePrediction()
    else:
        print('Oohhoo please select valid input!!!!!!!!! \n and \ntry again')