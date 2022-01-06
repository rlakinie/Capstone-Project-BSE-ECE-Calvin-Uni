# import onnx
# import argparse
# import onnx_tensorrt.backend as backend
# import numpy as np
# import time

# def main():
#     parser = argparse.ArgumentParser(description="Onnx runtime engine.")
#     parser.add_argument(
#         "--onnx", default="/home/quenny/Downloads/ICANN.onnx",
#         metavar="FILE",
#         help="path to onnx file",
#     )
#     parser.add_argument(
#         "--shape",
#         default="(32,3,256,256)",
#         help="input shape for inference",
#     )
#     args = parser.parse_args()
#     model = onnx.load(args.onnx)
#     engine = backend.prepare(model, device='CUDA:0')
#     shape_str = args.shape.strip('(').strip(')')
#     input_shape = []
#     for item in shape_str.split(','):
#         input_shape.append(int(item)) 
#     input_data = np.random.random(size=input_shape).astype(np.float32)
#     start = time.time()
#     cal = []
#     for i in range(2556):
#         output_data = engine.run(input_data)[0]
#         cal.append(time.time())
#     end = time.time()
#     total_time = end - start
#     print("Total Runtimetime {:.4f} seconds".format(total_time))
#     start = cal[10]
#     Per_time = ( end -start ) / 100.0
#     print("Per iter runtime: {:.4f} seconds".format(Per_time))
    
# if __name__ == "__main__":
#     print ("Usage: .... ")
#     print ("python tensorrt_run.py --onnx your.onnx --shape (1,3,112,112)")
#     main()

import torch
import torch.onnx
import torchvision
import torchvision.models as models
import sys



import torch.nn as nn
import torch.nn.functional as F
import os
import torchvision.models as models
import numpy as np 
import cv2



def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epocdummy_input = torch.randn(32, 3, 256, 256, device='cuda')

device = torch.device('cuda')


class ResNet(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        # Use a pretrained model
        self.network = models.resnet50(pretrained=True)
        # Replace last layer
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, 6)
    
    def forward(self, xb):
        return torch.sigmoid(self.network(xb))

    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

#Create some sample input in the shape this mnodel expects.
dummy_input = torch.randn(32, 3, 256, 256, device='cuda')
onnx_model_path = "/home/quenny/Downloads/ICAN.onnx"

device = torch.device('cuda')
model = ResNet()
model.eval()

sample_image = cv2.VideoCapture()

net = cv2.dnn.readNetFromONNX(onnx_model_path)
image=cv2.imread(sample_image)
blob=cv2.dnn.blobFromImage(image, 1.0/255, (224, 224), (0,0,0), swapRB=True, crop=False)
net.setInput(blob)
preds = net.forward()
biggest_pred_index = np.array(preds)[0].argmax()
print("Predicted class:", biggest_pred_index)

labels = [ ' Insert all 6 labesl here']

print("The class", biggest_pred_index "corresponds to ", labels[biggest_pred_index])

def predict_image(img, model):
    # Convert to a batch of 1
    xb = to_device(img.unsqueeze(0), device)
    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    prob, preds  = torch.max(yb, dim=1)
    # Retrieve the class label
    return dataset.classes[preds[0].item()]


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)