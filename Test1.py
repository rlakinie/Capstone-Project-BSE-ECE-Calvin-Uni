import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.data import random_split
from PIL import Image
from pathlib import Path
import numpy as np
import cv2
import time
import torchvision.transforms as T
from PIL import Image 



data_dir = '/home/ican/Downloads/wow/Garbage classification/Garbage classification'
classes = os.listdir(data_dir)

from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

transformations = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

dataset = ImageFolder(data_dir, transform = transformations)

#train_ds, val_ds, test_ds = random_split(dataset, [1593, 176, 758])



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
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch {}: train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch+1, result['train_loss'], result['val_loss'], result['val_acc']))


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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)

model = ResNet()
model.load_state_dict(torch.load('/home/ican/Downloads/newest_file', map_location="cuda:0"))
model.to(device)
model.eval()


def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=820,
    display_height=616,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )



def predict_image(img, model):
    # Convert to a batch of 1
    xb = to_device(img.unsqueeze(0), device)
    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    prob, preds  = torch.max(yb, dim=1)

    # Retrieve the class label
    return dataset.classes[preds[0].item()]

#img, label = test_ds[27]
#plt.imshow(img.permute(1, 2, 0))

def test_one():
    image = Image.open(Path('/home/ican/Downloads/test2/lacroix10.jpg'))
    img  = transformations(image)
    print('predicted', predict_image(img, model))
    return 

def transformses(trains):
    new_image = transformations(trains)
    return new_image




prev_frame_time = 0
next_frame_time = 0
cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
ret = True

clip = []
while(ret):
    # Capture frame-by-frame
    #window_handle = cv2.namedWindow('iCAN_Detection', cv2.WINDOW_AUTOSIZE)
    ret, frame = cap.read()

    gray = cv2.resize(frame, (256, 256))
    gray2 = cv2.cvtColor(gray, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(gray2)

    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time) 
    prev_frame_time = new_frame_time
    #frame2 = frame.astype('float32')
    #curr_img = cv2.resize(pil_image, (256, 256))
    #curr_img has been reiszew, apply PIL to it 


    now_img = transformses(pil_image)
    # converting the fps into integer 
    #
    fps = int(fps) 
  
    # converting the fps to string so that we can display it on frame 
    # by using putText function 
    fps = str(fps) 

    #img = transformations(frame)
    #print('predicted', predict_image(img, model))
    print('predicted', predict_image(now_img, model))

    #in between put somethng 
    cv2.putText(frame, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0))
    cv2.imshow('image', pil_image)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#test_one()xasd