import torch.nn as nn 
import torch.nn.functional as F

class FFD(nn.Module):
    def __init__(self, n_input, n_hidden, n_output=1):
        super(FFD, self).__init__()

        self.fc1 = nn.Linear(n_input,n_hidden)
        self.fc2 = nn.Linear(n_hidden,n_output)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.sigmoid(x)
        return x
    
