import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader 
from DataLoaderTrain import loadDataSubSample
from NeuralNets import NeuralNet, DeepNeuralNetwork
from torch.optim.lr_scheduler import ExponentialLR

class CustomDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        # Return the number of samples
        return len(self.inputs)

    def __getitem__(self, index):
        # Retrieve the input and target at the specified index
        x = self.inputs[index]
        y = self.targets[index]
        return x, y

def trainModel(feat_lst, targ_lst):


    exp = "v02"

    input, target, input_val, target_val = loadDataSubSample(feat_lst, targ_lst)

    dataset = CustomDataset(input, target)
    dataset_val = CustomDataset(input_val, target_val)

    #train_size = int(0.8 * len(dataset))
    #val_size = len(dataset) - train_size

    #train_data, val_data = random_split(dataset, [train_size, val_size])

    val_loader = DataLoader(dataset_val, batch_size=32, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    #model = SimpleNet()
    model = NeuralNet()
    input_size = 19
    hidden_sizes = [32,64,32]
    output_size = 1

    #model = DeepNeuralNetwork(input_size, hidden_sizes, output_size)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    #scheduler = ExponentialLR(optimizer, gamma=0.9) 
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 5, gamma = 0.5)
    epochs = 30
    best_valid_loss = float("inf")

    for epoch in range(epochs):

        model.train()
        for batch in dataloader:
            inputs, targets = batch            
            optimizer.zero_grad()        
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        #print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

        model.eval()
        val_loss = 0.0
        correct = 0.0
        total = 0.0

        with torch.no_grad():  # Disable gradient calculation for validation
            for batch in val_loader:
                inputs, targets = batch
                #inputs = inputs.view(-1, 28*28)  # Flatten images
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item()*targets.size(0)

                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)                
                #correct += (predicted == targets).sum().item()

        average_valid_loss = val_loss / total                               # Compute the mean loss for the epoch
        current_learning_rate = optimizer.param_groups[0]['lr']             # Extract the current learning rate
        #optimizer.step()                                                   # Update learning rate
        scheduler.step()

        #accuracy = correct / total
        print(f'Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss/len(val_loader)},LR: {current_learning_rate}')

        if average_valid_loss < best_valid_loss:            
            model_path = "/ec/res4/scratch/sbjb/Projects/CERISE/ObsOpData/NN/18GHz/Models/%s/mlp_dual_pol_training.pth"%exp
            # Save the model's state dictionary 
            torch.save(model.state_dict(), model_path)
            best_valid_loss = average_valid_loss

def main():

    feat_lst = [
        "ZS",        
        "FRAC_LAND_AND_SEA_WATER", 
        "Distance_to_footprint_center",
        "TG1_ga",
        "TG2_ga",
        "WG1_ga",
        "WG2_ga",
        "WGI1_ga",
        "WGI2_ga",
        "TS_ISBA",     
        "LAI_ga",
        "HSN_VEG1_ga",
        "HSN_VEG6_ga",
        "HSN_VEG12_ga",
        "RSN_VEG1_ga",
        "RSN_VEG6_ga",
        "RSN_VEG12_ga",         
        "WSN_T_ISBA",
        "DSN_T_ISBA"        
    ]

    targ_lst = [
        "AMSR2_BT18.7V",    
        "AMSR2_BT18.7H",    
    ]
    
    trainModel(feat_lst, targ_lst)

if __name__ == "__main__":
    
    main()