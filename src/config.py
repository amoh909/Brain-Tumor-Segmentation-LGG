IMAGE_SIZE = 256
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
NUM_EPOCHS = 1
EXPERIMENT_ID = "E6"
MODEL_TYPE = "unet" # "unet" or "attention_unet", note: "plain unet could be used by changing the import in train.py and evaluation.py and using the UNetplain model directly"
LOSS_TYPE = "bce"   # "bce" or "dice" or "bce_dice"

TRAIN_CSV = "data/processed/train.csv"
VAL_CSV = "data/processed/val.csv"
TEST_CSV = "data/processed/test.csv"
CHECKPOINTS_DIR = "outputs/checkpoints"
EVALUATION_DIR = "outputs/evaluation"
