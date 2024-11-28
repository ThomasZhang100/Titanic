
#%%
import torch
import torch.optim as optim
import torch.nn as nn
from Model import FFD
from DataProcessing import train_loader, test_loader
import matplotlib.pyplot as plt

model = FFD(12,12,1)
optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum=0.9)
criterion = nn.BCELoss()

epochs = 700
loss_data = []
for e in range(epochs):
    running_loss = 0
    for inputs, targets in train_loader:
        #print(targets.shape)
        optimizer.zero_grad()
        outputs = model(inputs)
        targets = targets.view(-1,1)
        loss = criterion(outputs,targets)
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
    loss_data.append(loss.item())

torch.save(model.state_dict(), "model.pt") 
           
#%%
plt.plot(range(1,epochs+1),loss_data)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show();

print("Training loss:", running_loss/len(train_loader))


# %%

'''
model.eval()
with torch.no_grad():
    validation_loss = 0
    for inputs, targets in test_loader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        validation_loss+=loss 
    validation_loss = validation_loss / test_loader.shape[0]
print("validation_loss:", validation_loss)
'''




