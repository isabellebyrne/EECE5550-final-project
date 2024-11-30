from data import CustomPairwiseDataset
from model import OverFeat
import torch
from torch.utils.data import DataLoader,ConcatDataset,random_split
from torchsummary import summary
import torch.nn.functional as F
from tqdm import tqdm

#create dataloaders for CityCentre and NewCollege datasets
def createCombinedDataset(image_dirs, gt_matrix_paths):

    datasets = []
    for image_dir, gt_matrix in zip(image_dirs, gt_matrix_paths):
        datasets.append(CustomPairwiseDataset(image_dir, gt_matrix))
    
    combined_dataset = ConcatDataset(datasets)

    return combined_dataset

def contrastive_loss(features1, features2, label, margin=1.0):
    distance = torch.norm(features1 - features2, p=2, dim=1)
    loss = label * distance.pow(2) + (1 - label) * F.relu(margin - distance).pow(2)
    return loss.mean()

def train_step(model, dataloader, optimizer, device, log_interval=10):
    model.train()
    total_loss = 0
    for batch_idx, (img1, img2, labels) in tqdm(enumerate(dataloader), desc='Training', total=len(dataloader), ncols=200):  
        #move data to appropriate device
        img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

        features1 = model(img1)
        features2 = model(img2)

        loss = contrastive_loss(features1, features2, labels)

        #backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (batch_idx + 1) % log_interval == 0:
                print(f"Validation Batch {batch_idx + 1}/{len(dataloader)} - Loss: {loss.item():.4f}, "
                      f"Avg Loss: {total_loss / (batch_idx + 1):.4f}")
                
        if batch_idx == 10:
            break

    return total_loss / len(dataloader)

def validation_step(model, dataloader, device, log_interval=10):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_idx, (img1, img2, labels) in tqdm(enumerate(dataloader), desc='Validation', total=len(dataloader), ncols=200):
            #move data to appropriate device
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

            features1 = model(img1)
            features2 = model(img2)

            loss = contrastive_loss(features1, features2, labels)

            total_loss += loss.item()

            if (batch_idx + 1) % log_interval == 0:
                print(f"Validation Batch {batch_idx + 1}/{len(dataloader)} - Loss: {loss.item():.4f}, "
                      f"Avg Loss: {total_loss / (batch_idx + 1):.4f}")

    return total_loss / len(dataloader)

image_dirs = ['data/CityCentre_Images',
              'data/NewCollege_Images']
gt_matrix_paths = ['data/CityCentreTextFormat.txt',
                   'data/NewCollegeTextFormat.txt']

combined_dataset = createCombinedDataset(image_dirs, gt_matrix_paths)

#Split dataset into train and testsets
train_size = int(0.8 * len(combined_dataset))
validation_size = len(combined_dataset) - train_size
train_dataset, validation_dataset = random_split(combined_dataset, [train_size, validation_size])

#print length of combined dataset
print('Length of Combined Dataset:',len(combined_dataset))

# Device setup for compatibility with Mac, CUDA, or CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print('Using device:', device)

model = OverFeat()
print(summary(model, input_size=(3, 221, 221)))
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 10
batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

for epoch in range(epochs):
    train_loss = train_step(model, train_dataloader, optimizer, device)
    validation_loss = validation_step(model, validation_dataloader, device)
    print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss}, Validation Loss: {validation_loss}')



