#%%

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

train_data = pd.read_csv('train.csv')
test_data_raw = pd.read_csv('test.csv')

sns.boxplot(x='Pclass',y='Age',data=train_data,palette='winter')

train_data = pd.get_dummies(train_data, columns=['Sex', 'Pclass', 'Embarked'])
test_data = pd.get_dummies(test_data_raw, columns=['Sex', 'Pclass', 'Embarked'])
train_data = train_data.drop(['PassengerId','Name',"Cabin","Ticket"], axis=1)
test_data = test_data.drop(['PassengerId','Name',"Cabin","Ticket"], axis=1)
print(train_data.info())
print(test_data.info())

plt.figure(figsize=(12, 7))
sns.heatmap(train_data.isnull(),yticklabels=False,cbar=False,cmap='viridis')

def fillAge(row):
	if pd.isnull(row['Age']):
		if row['Pclass_1']==1:
			return 37
		elif row['Pclass_2']==1:
			return 29
		else:
			return 25
	else:
		return row['Age']
# %%
train_data['Age']=train_data.apply(fillAge,axis=1)
test_data['Age']=test_data.apply(fillAge,axis=1)
print(train_data.head())
plt.figure(figsize=(12, 7))
sns.heatmap(train_data.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# %%
features_to_normalize = ['Age','Fare']
scaler = StandardScaler()
train_data[features_to_normalize] = scaler.fit_transform(train_data[features_to_normalize])
test_data[features_to_normalize] = scaler.transform(test_data[features_to_normalize])
print(train_data.head())
print(test_data.head())
# %%
class datamap(torch.utils.data.Dataset):
	def __init__(self, df):
		data = torch.from_numpy(df.values.astype(np.float32))
		self.inputdata = data[:,1:]
		self.inputdata = self.inputdata.float()
		self.targets = data[:,0]
	
	def __len__(self):
		return self.targets.shape[0]
	
	def __getitem__(self, index):
		return self.inputdata[index], self.targets[index]

train_data_map = datamap(train_data)
test_data_map = datamap(test_data)
train_loader = torch.utils.data.DataLoader(train_data_map,batch_size=200)
test_loader = torch.utils.data.DataLoader(test_data_map,batch_size=200)