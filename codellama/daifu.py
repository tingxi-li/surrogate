import os
import pdb
import json
import torch
import daifu
import random
import requests
from PIL import Image

random.seed(0)
torch.manual_seed(0)


# Load model directly
from transformers import DetrImageProcessor, DetrForObjectDetection

processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

def cell_1():
    tensor_j = torch.randn(224, 224)
    cell_2(tensor_j)
    
def cell_2(tensor_j):
    cell_3(tensor_j)
    
def cell_3(tensor_j):
    cell_4(tensor_j)
    
def cell_4(tensor_j):
    
    tensor_k = torch.randn(8, 8)
        
    res = tensor_j * tensor_k

@daifu.transform()
def main(x):
    print("=== now start ===")
    print("x value: ", x)

    for i in range(x):
        cell_1()
    pass

if __name__ == "__main__":
    main(10)
