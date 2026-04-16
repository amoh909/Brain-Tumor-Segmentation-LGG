IMAGE_SIZE = 256
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
NUM_EPOCHS = 20
EXPERIMENT_ID = "E11"
MODEL_TYPE = "attention_unet" # "unet" or "attention_unet"
LOSS_TYPE = "bce_dice"   # "bce" or "dice" or "bce_dice"

TRAIN_CSV = "data/processed/train.csv"
VAL_CSV = "data/processed/val.csv"
TEST_CSV = "data/processed/test.csv"
CHECKPOINTS_DIR = "outputs/checkpoints"
EVALUATION_DIR = "outputs/evaluation"
