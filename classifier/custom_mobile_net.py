import torch.nn as nn
import torch
import torchvision.models as models
from preprocessor.custom_preprocessor import CustomPreprosessor

# 모델 정의
class CustomMobileNet(nn.Module):
    def __init__(self, input = 512):
        super(CustomMobileNet, self).__init__()
        self.input_preprocessor = CustomPreprosessor(input)
        self.mobilenet = models.mobilenet_v3_small(pretrained=True)
        self.mobilenet.classifier[3] = nn.Linear(self.mobilenet.classifier[3].in_features, 4)
        
    def forward(self, x):
        x = self.input_preprocessor(x)
        x = self.mobilenet(x)
        return x

    def load_pretrained(self, path):
        self.load_state_dict(torch.load(path))

    def set_transfer_learn(self):
        # 특정 레이어만 학습하도록 설정
        for param in self.parameters():
            param.requires_grad = False

        for param in self.input_preprocessor.parameters():
            param.requires_grad = True

        for param in self.mobilenet.classifier.parameters():
            param.requires_grad = True
        
        return self
        
            
# 학습 함수
def train_model(model, train_loader, criterion, optimizer, device, num_epochs=25):
    model.train()
    train_loss = 0
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).float()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        train_loss = epoch_loss
        #print(f'Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.4f}')
    return train_loss

# 검증 함수
def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device).float()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
    val_loss /= len(val_loader.dataset)
    #print(f'Validation Loss: {val_loss:.4f}')
    return val_loss
