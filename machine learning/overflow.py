#!/usr/bin/env python
# coding: utf-8

# # Getting Data in Form of List of Dictionary

# DONT RERUN THIS UNLESS ABSOLUTELY NESSESARY

# 

# In[212]:



# In[ ]:


import torch, torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from torchsummary import summary
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import os

from PIL import Image

img_directory = r"/Users/neerajakulkarni/Documents/WADABA"
images = os.listdir(img_directory)
data = []




def parse_name(fn):
    print(fn[0:4])
    ptype = int(fn[fn.find('a')+1 : fn.find('a') + 3]) 
    ptype = ptype - 3 if ptype == 5 or ptype == 6 or ptype == 7 else ptype -1
    color = fn[fn.find('b')+1 : fn.find('b') + 3]
    light = fn[fn.find('c')+1 : fn.find('c') + 2]
    deform = fn[fn.find('d')+1 : fn.find('d') + 2]
    dirt = fn[fn.find('e')+1 : fn.find('e') + 2]
    sl = fn[fn.find('f')+1 : fn.find('f') + 2]
    ring = fn[fn.find('g')+1 : fn.find('g') + 2]
    rand_pos = fn[fn.find('h')+1 : fn.find('h') + 2]
    return ptype, color, light, deform, dirt, sl, ring, rand_pos



for img in images:
    if img[0].isnumeric():
        current_img = Image.open((os.path.join(img_directory, img)))
        ptype, color, light, deform, dirt, sl, ring, rand_pos = parse_name(img)
        img_arr = np.asarray(current_img.resize((224,224)))
        img_data = {'plastic_type': ptype, 'color': color, 'light': light, 'deformation':deform,
                    'dirtiness':dirt, "screw cap or lid": sl, 'ring':ring, 'random position': rand_pos, 'image': img_arr}
        current_img.close()
        data.append(img_data)
    
       


# # Converting Data to DataFrame

# In[214]:




# In[215]:


import pandas as pd
df = pd.DataFrame(data)

Image.fromarray(df['image'][0])

df['plastic_type'].value_counts()



# In[216]:


from sklearn.model_selection import train_test_split
import pandas as pd
df = df[df['plastic_type'] != 6]

PET = df[df['plastic_type'] == 0]
PE_HD = df[df['plastic_type'] == 1]
PP = df[df['plastic_type'] == 2]
PS = df[df['plastic_type'] == 3]


# In[217]:


def split_all_data(df):
  X = df['image']
  y = df['plastic_type']

  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42) #42

  return X_train, X_test, y_train, y_test


# In[218]:


PET_x_train, PET_x_test, PET_y_train, PET_y_test = split_all_data(PET)
PE_HD_x_train, PE_HD_x_test, PE_HD_y_train, PE_HD_y_test = split_all_data(PE_HD)
PP_x_train, PP_x_test, PP_y_train, PP_y_test = split_all_data(PP)
PS_x_train, PS_x_test, PS_y_train, PS_y_test = split_all_data(PS)



# In[219]:


X_train = pd.concat([PET_x_train, PE_HD_x_train, PP_x_train, PS_x_train])
X_test = pd.concat([PET_x_test, PE_HD_x_test, PP_x_test, PS_x_test])

y_train = pd.concat([PET_y_train, PE_HD_y_train, PP_y_train, PS_y_train])
y_test = pd.concat([PET_y_test, PE_HD_y_test, PP_y_test, PS_y_test])



# In[220]:


X_test = pd.concat([PET_x_test, PE_HD_x_test, PP_x_test, PS_x_test])
y_test = pd.concat([PET_y_test, PE_HD_y_test, PP_y_test, PS_y_test])

X_valid, X_test, y_valid, y_test = train_test_split(
      X_test, y_test, test_size=0.5, random_state=42)


# In[221]:


train_df = pd.concat([X_train, y_train], axis=1)
valid_df = pd.concat([X_valid, y_valid], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)


# In[ ]:





# In[222]:


from torch.utils.data import DataLoader, Dataset
class CustomImageDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_arr = self.df['image'].iloc[idx]
        image = Image.fromarray(image_arr)
        label = self.df['plastic_type'].iloc[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


# In[223]:


train_data = CustomImageDataset(train_df,transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]))
valid_data = CustomImageDataset(valid_df,transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]))

test_data = CustomImageDataset(valid_df,transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]))


train_data_loader = DataLoader(train_data, batch_size=16, shuffle=True)
valid_data_loader = DataLoader(valid_data, batch_size=16, shuffle=True)
test_data_loader = DataLoader(test_data, batch_size=16, shuffle=True)



# # Starting out our Models

# Modifying The CNNS
# 

# In[224]:


#alexnet
alexnet = models.alexnet(pretrained=True)

for param in alexnet.parameters():
    param.requires_grad = False

alexnet.classifier[6] = nn.Linear(4096, 4)#Changing final layer
alexnet.classifier.add_module("7", nn.LogSoftmax(dim = 1)) #adding a layer to classify


#resnet50 takes pretty long
resnet50 = models.resnet50(pretrained=True)

for param in resnet50.parameters():
    param.requires_grad = False

resnet50.fc = nn.Linear(2048, 4)
resnet50.add_module("7", nn.LogSoftmax(dim = 1))


#VGG SUPER SLOW

VGG = models.vgg19()

for param in VGG.parameters():
    param.requires_grad = False


VGG.classifier[6] = nn.Linear(4096, 4)#Changing final layer
VGG.classifier.add_module("7", nn.LogSoftmax(dim = 1)) #adding a layer to classify

# inception 
inception = models.inception_v3()

for param in inception.parameters():
    param.requires_grad = False

inception.AuxLogits.fc = nn.Linear(768, 4) 
inception.fc = nn.Linear(2048, 4)  
inception.add_module("7", nn.LogSoftmax(dim=1))  


# In[225]:


loss_func = nn.NLLLoss()
optimizer = optim.Adam(alexnet.parameters(), lr=0.001)
optimizer


# In[226]:


def train_and_validate(model, loss_criterion, optimizer, epochs=25):
    '''
    Function to train and validate
    Parameters
        :param model: Model to train and validate
        :param loss_criterion: Loss Criterion to minimize
        :param optimizer: Optimizer for computing gradients
        :param epochs: Number of epochs (default=25)
  
    Returns
        model: Trained Model with best validation accuracy
        history: (dict object): Having training loss, accuracy and validation loss, accuracy
    '''
    
    start = time.time()
    history = []
    best_acc = 0.0

    for epoch in range(epochs):
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch+1, epochs))
        
        # Set to training mode
        model.train()
        
        # Loss and Accuracy within the epoch
        train_loss = 0.0
        train_acc = 0.0
        
        valid_loss = 0.0
        valid_acc = 0.0
        
        for i, (inputs, labels) in enumerate(train_data_loader):
           

            inputs = inputs.to(device)
           # labels = labels.to(device)

        
            # Clean existing gradients
            optimizer.zero_grad()
            
            # Forward pass - compute outputs on input data using the model
            outputs = model(inputs)
            
            # Compute loss
            loss = loss_criterion(outputs, labels)
            
            # Backpropagate the gradients
            loss.backward()
            
            # Update the parameters
            optimizer.step()
            
            # Compute the total loss for the batch and add it to train_loss
            train_loss += loss.item() * inputs.size(0)
            
            # Compute the accuracy
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))
            
            # Convert correct_counts to float and then compute the mean
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            
            # Compute total accuracy in the whole batch and add to train_acc
            train_acc += acc.item() * inputs.size(0)
            
            print("Batch number: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}".format(i, loss.item(), acc.item()))

            
        # Validation - No gradient tracking needed
        with torch.no_grad():

            # Set to evaluation mode
            model.eval()

            # Validation loop
            for j, (inputs, labels) in enumerate(valid_data_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward pass - compute outputs on input data using the model
                outputs = model(inputs)

                # Compute loss
                loss = loss_criterion(outputs, labels)

                # Compute the total loss for the batch and add it to valid_loss
                valid_loss += loss.item() * inputs.size(0)

                # Calculate validation accuracy
                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))

                # Convert correct_counts to float and then compute the mean
                acc = torch.mean(correct_counts.type(torch.FloatTensor))

                # Compute total accuracy in the whole batch and add to valid_acc
                valid_acc += acc.item() * inputs.size(0)

                #print("Validation Batch number: {:03d}, Validation: Loss: {:.4f}, Accuracy: {:.4f}".format(j, loss.item(), acc.item()))
            
        # Find average training loss and training accuracy
        avg_train_loss = train_loss/len(train_data) 
        avg_train_acc = train_acc/len(train_data) 

        # Find average training loss and training accuracy
        avg_valid_loss = valid_loss/len(test_data)  
        avg_valid_acc = valid_acc/len(test_data) 

        history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])
                
        epoch_end = time.time()
    
        print("Epoch : {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tValidation : Loss : {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(epoch+1, avg_train_loss, avg_train_acc*100, avg_valid_loss, avg_valid_acc*100, epoch_end-epoch_start))
        
        # Save if the model has best accuracy till now
        #torch.save(model, dataset+'_model_'+str(epoch)+'.pt')
            
    return model, history


# In[227]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_epochs = 1
#trained_model, history = train_and_validate(alexnet, loss_func, optimizer, num_epochs)
trained_model, history = train_and_validate(alexnet, loss_func, optimizer, num_epochs)

torch.save(history, 'model' +'_history.pt')
torch.save(trained_model.state_dict(), 'model.pt')


# In[228]:


history = np.array(history)
plt.plot(history[:,0:2])
plt.legend(['Tr Loss', 'Val Loss'])
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.ylim(0,1)
#plt.xlim(-1,num_epochs)
plt.savefig('_loss_curve.png')
plt.show()


# In[229]:


history


# In[230]:


plt.plot(history[:,2:4])
plt.legend(['Tr Accuracy', 'Val Accuracy'])
plt.xlabel('Epoch Number')
plt.ylabel('Accuracy')
plt.ylim(0,1)
plt.savefig('_accuracy_curve.png')
plt.show()


# In[231]:


def img_transform(img):
    img_transformations = transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        ])
    return img_transformations(img)


# In[232]:


def process_image(fn):
    img = Image.open(fn).convert('RGB')
    img = img.resize((224,224))

    test_img_tensor = img_transform(img)

    return test_img_tensor


# In[233]:


#def test_model(input):
 #   img = process_image(input)
  #  return alexnet(img)


# In[234]:


import torch, torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from torchsummary import summary
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import os

from PIL import Image

model = alexnet

model.load_state_dict(torch.load('model.pt', weights_only=True))
model.eval()



#def test_model(input):
#    img = process_image(input)
 #   return model(img)

#test_model('img.png')


# In[245]:


def predict(model, test_image_name):
    '''
    Function to predict the class of a single test image
    Parameters
        :param model: Model to test
        :param test_image_name: Test image

    '''
    transform = transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ])

    test_image = Image.open(test_image_name).convert('RGB')
    plt.imshow(test_image)
    
    test_image_tensor = transform(test_image)

    if torch.cuda.is_available():
        test_image_tensor = test_image_tensor.view(1, 3, 224, 224).cuda()
    else:
        test_image_tensor = test_image_tensor.view(1, 3, 224, 224)
    
    with torch.no_grad():
        model.eval()
        # Model outputs log probabilities
        out = model(test_image_tensor)
        print(torch.argmax(out).item())

        ps = torch.exp(out)
        topk = ps.topk(4, dim=1)

        #scores = topk.values.numpy()
        #predictions = topk.indices.numpy()

        #max_score_index = np.argmax(scores)
        #max_pred = predictions[max_score_index]

    # print(max_pred)

        results = topk.values.cpu().numpy()[0]
        #print(results)
        #print(np.where( results == max(results))[0][0])


# In[241]:


def run_test(model, loss_criterion):
     
   
    model.eval()
    valid_loss = 0.0
    valid_acc = 0.0
    with torch.no_grad():

            # Set to evaluation mode
            

            # Validation loop
            for j, (inputs, labels) in enumerate(valid_data_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward pass - compute outputs on input data using the model
                outputs = model(inputs)

                # Compute loss
                loss = loss_criterion(outputs, labels)

                # Compute the total loss for the batch and add it to valid_loss
                valid_loss += loss.item() * inputs.size(0)

                # Calculate validation accuracy
                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))

                # Convert correct_counts to float and then compute the mean
                acc = torch.mean(correct_counts.type(torch.FloatTensor))

                # Compute total accuracy in the whole batch and add to valid_acc
                valid_acc += acc.item() * inputs.size(0)

                #print("Validation Batch number: {:03d}, Validation: Loss: {:.4f}, Accuracy: {:.4f}".format(j, loss.item(), acc.item()))
            

    avg_valid_loss = valid_loss/len(test_data)  
    avg_valid_acc = valid_acc/len(test_data) 

    return avg_valid_loss, avg_valid_acc


# In[242]:


run_test(alexnet, loss_func)


# TESTING
# 

# In[246]:


#WRONG



# In[247]:


#RIGHT
# In[250]:


#predict(model, 't2.png')
#predict(model, 't1.png')
