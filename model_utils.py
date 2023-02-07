import torch
import torch.nn as nn
from torchvision.transforms import transforms
from PIL import Image
from torch.autograd import Variable


classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

print ("All Libs loaded")

#cnn network

class CNNNet(nn.Module):
    def __init__(self, num_classes=6):
        super(CNNNet, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=12)
        self.relu = nn.ReLU()
        
        self.pool = nn.MaxPool2d(kernel_size=2)
        
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=20, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=20)
        self.relu2 = nn.ReLU()
        
        self.conv3 = nn.Conv2d(in_channels=20, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=32)
        self.relu3 = nn.ReLU()
        
        self.fc = nn.Linear(in_features=32*75*75, out_features=num_classes)
        
    def forward(self, input):
        output = self.conv1(input)
        output = self.bn1(output)
        output = self.relu(output)
        output = self.pool(output)
        output = self.conv2(output)
        output = self.bn2(output)
        output = self.relu2(output)
        
        output = self.conv3(output)
        output = self.bn3(output)
        output = self.relu3(output)
        
        output = output.view(-1, 32*75*75)
        
        output = self.fc(output)
        return output

map_location=torch.device('cpu')
checkpoint = torch.load('./model/best_checkpoint.model', map_location=map_location)
model = CNNNet(num_classes=6)
model.load_state_dict(checkpoint)

model.eval()


transformer = transforms.Compose([
    transforms.Resize((150,150)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                        [0.5,0.5,0.5])
])

def prediction(image, transformer):
    
    # image = Image.open(img_path)
    image_tensor = transformer(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    
    if torch.cuda.is_available():
        image_tensor.cuda()
    
    input=Variable(image_tensor)
    
    output = model(input)
    index = output.data.numpy().argmax()
    
    pred = classes[index]
    return pred

print ("Code completed")


