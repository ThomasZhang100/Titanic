import torch
import numpy as np
import pandas as pd
from Model import FFD
from DataProcessing import test_data, test_data_raw

model = FFD(12,12,1)
model.load_state_dict(torch.load('model.pt'))
model.eval()

inputs = torch.from_numpy(test_data.values.astype(np.float32))

print(test_data.head())

with torch.no_grad():
    predictions = torch.round(model(inputs)).type(torch.int)

output = pd.DataFrame({'PassengerId': test_data_raw.PassengerId, 'Survived': predictions.flatten()})
output.to_csv('submission.csv', index=False)