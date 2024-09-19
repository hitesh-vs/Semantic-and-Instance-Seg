import os

# Paths
HOME_PATH = os.path.expanduser("~/test")
JOB_ID = "semSeg1"
MODEL_NAME = "windowSegSeg"
IMG_DIR = "~/test/data/imgs"
MASK_DIR = "~/test/data/masks"  
OUT_PATH = "~/test/outputs/windowseg"

# Other parameters
JOB_FOLDER = os.path.join(OUT_PATH, JOB_ID)
TRAINED_MDL_PATH = os.path.join(JOB_FOLDER, "parameters")
BATCH_SIZE = 128
LR = 1e-4
LOG_BATCH_INTERVAL = 1
NUM_WORKERS = 8

# Add image size for resizing in dataloader
IMG_SIZE = (256, 256)
