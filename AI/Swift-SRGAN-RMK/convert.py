"""
import torch
import torch.onnx
import torchvision
import torchvision.models as models
import sys

onnx_model_path = "./test/TrainedModels/"

model = torch.jit.load("./test/TrainedModels/optimized_model_16.pt")
#model = model.cuda()
model.eval()
model.cpu()
# Create some sample input in the shape this model expects 
# This is needed because the convertion forward pass the network once 
dummy_input = torch.randn(1, 3, 224, 224)
#torch.onnx.export(model, dummy_input, onnx_model_path, verbose=True)
onnx_program = torch.onnx.export(model, dummy_input, onnx_model_path, verbose=True)
onnx_program.save("my_image_classifier.onnx")
"""

import torch
import torch.onnx
from torchvision.models import resnet18  # You can replace this with your own PyTorch model
import tensorflow as tf
from onnx_tf.backend import prepare
import tf2onnx
import onnx

# Step 1: Load your PyTorch model
model = torch.jit.load("./test/TrainedModels/optimized_model_16.pt")
dummy_input = torch.randn(1, 3, 224, 224)  # Adjust the input size based on your model's input requirements

# Step 2: Export the PyTorch model to ONNX format
onnx_path = "model.onnx"
torch.onnx.export(model.cpu(), dummy_input, onnx_path, verbose=True)

# Step 3: Convert the ONNX model to TensorFlow format (.pb)
onnx_model = onnx.load(onnx_path)
tf_rep = tf2onnx.convert.from_onnx_file(onnx_path)
tf_rep.export_graph("model.pb")

print("Conversion completed successfully.")