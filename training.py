from data import createCombinedDataset
from model import OverFeat
import torch
from torch.utils.data import DataLoader,random_split
from torchsummary import summary
import torch.nn.functional as F
from tqdm import tqdm
from argparse import ArgumentParser
import re
import os

def contrastive_loss(distances, label, margin=1.0):
    loss = label * distances.pow(2) + (1 - label) * F.relu(margin - distances).pow(2)
    return loss.mean()

def calculate_accuracy(distances, labels, threshold=0.5):
    normalized_distances = (distances - distances.min()) / (distances.max() - distances.min())
    predictions = (normalized_distances <= threshold).float()
    correct_predictions = (predictions == labels).sum().item()
    accuracy = correct_predictions / labels.size(0)
    return accuracy

def train_step(model, dataloader, optimizer, device, log_interval=500):
    model.train()
    total_loss = 0
    total_accuracy = 0
    for batch_idx, (img1, img2, labels) in tqdm(enumerate(dataloader), desc='Training', total=len(dataloader), ncols=200):  
        #move data to appropriate device
        img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
        features1 = model(img1)
        features2 = model(img2)

        distances = torch.norm(features1 - features2, p=2, dim=1)
        loss = contrastive_loss(distances, labels)

        #backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate batch accuracy
        accuracy = calculate_accuracy(distances, labels)

        total_loss += loss.item()
        total_accuracy += accuracy

        if (batch_idx + 1) % log_interval == 0:
                print(f"Training Batch {batch_idx + 1}/{len(dataloader)} - Loss: {loss.item():.4f}, "
                      f"Avg Loss: {total_loss / (batch_idx + 1):.4f}, Accuracy: {accuracy:.4f}, "
                      f"Avg Accuracy: { total_accuracy / (batch_idx + 1):.4f}")
    
    return total_loss / len(dataloader), total_accuracy / len(dataloader)

def validation_step(model, dataloader, device, log_interval=250):
    model.eval()
    total_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for batch_idx, (img1, img2, labels) in tqdm(enumerate(dataloader), desc='Validation', total=len(dataloader), ncols=200):
            #move data to appropriate device
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

            features1 = model(img1)
            features2 = model(img2)

            distances = torch.norm(features1 - features2, p=2, dim=1)
            loss = contrastive_loss(distances, labels)
            accuracy = calculate_accuracy(distances, labels)

            total_loss += loss.item()
            total_accuracy += accuracy
            
            if (batch_idx + 1) % log_interval == 0:
                print(f"Validation Batch {batch_idx + 1}/{len(dataloader)} - Loss: {loss.item():.4f}, "
                      f"Avg Loss: {total_loss / (batch_idx + 1):.4f}, Accuracy: {accuracy:.4f}, "
                      f"Avg Accuracy: { total_accuracy / (batch_idx + 1):.4f}")

    return total_loss / len(dataloader), total_accuracy / len(dataloader)

def get_epoch_num(filename):
    pattern = r'checkpoint_(\d+)\.pth'
    match = re.search(pattern, filename)
    
    if match:
        return int(match.group(1))
    else:
        raise ValueError("Filename does not match expected pattern")

if __name__ == '__main__':

    parser = ArgumentParser(description='OverFeat Training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--load_weights_path', type=str, default=None, help='checkpoint path to load model weights')
    parser.add_argument('--save_weights_dir', type=str, default='checkpoint.pth', help='dir to save model weights')
    args = parser.parse_args()

    image_dirs = ['data/CityCentre_Images',
                'data/NewCollege_Images']
    gt_matrix_paths = ['data/CityCentreTextFormat.txt',
                    'data/NewCollegeTextFormat.txt']

    combined_dataset = createCombinedDataset(image_dirs, gt_matrix_paths)

    #Split dataset into train and testsets
    train_size = int(0.8 * len(combined_dataset))
    validation_size = len(combined_dataset) - train_size
    train_dataset, validation_dataset = random_split(combined_dataset, [train_size, validation_size], 
                                                     generator=torch.Generator().manual_seed(42))

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
    epochs = args.epochs
    batch_size = args.batch_size

    if device.type == "mps":
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    else:
        #change num_workers based on number of cpu cores available
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, prefetch_factor=4, pin_memory=True)
        validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=8, prefetch_factor=4, pin_memory=True)

    if args.load_weights_path:
        #get filename from path
        abspath = os.path.abspath(args.load_weights_path)
        print("abspath:", abspath)
        checkpoint = torch.load(abspath)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        #get filename abs_path
        filename = os.path.basename(abspath)
        current_epoch = get_epoch_num(filename)

        training_losses = checkpoint['training_losses']
        validation_losses = checkpoint['validation_losses']
        training_accs = checkpoint['training_accs']
        validation_accs = checkpoint['validation_accs']

        print("last training and validation acc:", training_accs[-1], validation_accs[-1])
        print("last training and validation losses:", training_losses[-1], validation_losses[-1])
        #current epoch till chkpt 4 is 1, and chkpt 5 is 4 (error in code) but weights are correct
        print('Current Epoch Number:', current_epoch)
        
        print('Checkpoint loaded successfully.')

    else:
        print('No checkpoints to load')
        current_epoch = 0

        training_losses = []
        validation_losses = []
        training_accs = []
        validation_accs = []

    os.makedirs(args.save_weights_dir, exist_ok=True)
    save_weights_dir = os.path.abspath(args.save_weights_dir)

    print("save_weights_dir:", save_weights_dir)

for epoch in range(current_epoch,epochs+1):
    train_loss, train_accuracy = train_step(model, train_dataloader, optimizer, device)
    validation_loss, validation_accuracy = validation_step(model, validation_dataloader, device)
    print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss}, Train Accuracy: {train_accuracy} Validation Loss: {validation_loss}, Validation Accuracy: {validation_accuracy}')

    training_losses.append(train_loss)
    validation_losses.append(validation_loss)
    training_accs.append(train_accuracy)
    validation_accs.append(validation_accuracy)
    
    checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch + 1, 
    'training_losses': training_losses,
    'validation_losses': validation_losses,
    'training_accs': training_accs,
    'validation_accs': validation_accs
    }

    torch.save(checkpoint, f'{save_weights_dir}/checkpoint_{epoch+1}.pth')



