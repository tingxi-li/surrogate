TRAIN_SIZE = 0.9
VAL_SIZE = 0.05
TEST_SIZE = 0.05

RM_COLS = {
    "completion": ["prompt", "response"],
    "dpo": ["prompt", "chosen", "rejected"],
    "rlhf": ["prompt", "response"]
}