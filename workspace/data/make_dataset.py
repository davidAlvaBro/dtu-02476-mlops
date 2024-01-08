import torch

def create_dataset():
    # Get the data and process it
    # Data path 
    data_path = "data/raw"
    
    # Load raw test data
    test_images = torch.load(f"{data_path}/test_images.pt")
    test_targets = torch.load(f"{data_path}/test_target.pt")
    
    # Load raw train data
    train_images = torch.Tensor([])
    train_targets = torch.Tensor([])
    for i in range(6): 
        images = torch.load(f"{data_path}/train_images_{i}.pt")
        targets = torch.load(f"{data_path}/train_target_{i}.pt")
        train_images = torch.cat((train_images, images))
        train_targets = torch.cat((train_targets, targets))
        
    # Normalize the images (let mean be 0 and std be 1)
    train_mean = train_images.mean()
    train_std = train_images.std()
    train_images = (train_images - train_mean) / train_std
    test_images = (test_images - train_mean) / train_std
    
    # Unsqueeze the images to add a channel dimension
    train_images = train_images.unsqueeze(1)
    test_images = test_images.unsqueeze(1)
    
    # Save processed data
    processed_data_path = "data/processed"
    torch.save(test_images, f"{processed_data_path}/test_images.pt")
    torch.save(test_targets, f"{processed_data_path}/test_target.pt")
    torch.save(train_images, f"{processed_data_path}/train_images.pt")
    torch.save(train_targets, f"{processed_data_path}/train_target.pt")
    
    # I did it 
    print("Data processed successfully!")
    

if __name__ == '__main__':
    create_dataset()