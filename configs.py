import torch

class Configs():
    # device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # training hyperparameters
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0001
    EPOCHS = 1000
    WEIGHT_DECAY = 1e-5

    # model hyperparameters
    IMG_HEIGHT = 32

    # CTC parameters
    BLANK_LABEL = "-"

    # paths
    DATA_FOLDER = "data/sentences"
    LABEL_FILE = "data/ascii/sentences.txt"
