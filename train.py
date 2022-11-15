from get_input_args import get_input_args_for_train
from check_arch_and_hidden_units import check_arch_and_hidden_units
from load_model_and_classifier import load_model_and_classifier
from create_dataloaders import create_dataloaders

import torch
from torch import nn
from torch import cuda
from torch import optim


def main():
    
    # Getting the command line arguments
    print('Getting Commanding Line Arguments ...')
    in_arg = get_input_args_for_train()
    # print(in_arg)
    
    # Get the values from the command line arguments
    data_dir = in_arg.data_dir
    use_gpu = in_arg.gpu
    arch = in_arg.arch
    epochs = in_arg.epochs
    learning_rate = in_arg.learning_rate
    hidden_units = in_arg.hidden_units
    checkpoint_dir = in_arg.save_dir
    
    # Checking the model and hiden layers
    checked_arch_and_hidden_units = check_arch_and_hidden_units(arch, hidden_units)
    
    if checked_arch_and_hidden_units:
        # Setting default device
        device = 'cpu'

        # Check GUP Available if the user wants to use GPU
        if use_gpu:
            is_gpu_available = torch.cuda.is_available()
            if is_gpu_available:
                device = 'cuda'
                print('Running in GPU mode ...')
            else:
                print('GPU unavailable. Running in CPU mode ....')
        else:
            print('Running in CPU mode ...')
            
        # Load the model
        model = load_model_and_classifier(arch, hidden_units)
        model = model.to(device)
        # print(model)
        
        # Error function
        criterion = nn.NLLLoss()

        # Optimizer for the model
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
        
        # Load and create train, test and validation dataloaders class_to_idx
        class_to_idx, train_dataloaders, valid_dataloaders, test_dataloaders = create_dataloaders(data_dir)
        
        # Train the model
        print("Training the Model")
        model = train_model(model, train_dataloaders, valid_dataloaders, criterion, optimizer, epochs, device)
        print("---Training the Model Completed.---")
        
        save_checkpoint(arch, model, checkpoint_dir, optimizer, class_to_idx)
        print("---Successfully saved the checkpoint.---")
           
        
def train_model(model, train_dataloaders, valid_dataloaders, criterion, optimizer, epochs, device):
    steps = 0
    running_loss = 0
    print_every = 10

    for epoch in range(epochs):
        for inputs, labels in train_dataloaders:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()

                with torch.no_grad():
                    for inputs, labels in valid_dataloaders:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        test_loss += batch_loss.item()

                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs} |"
                      f"Training loss: {running_loss/print_every:.3f} | "
                      f"Validation loss: {test_loss/len(valid_dataloaders):.3f} | "
                      f"Validation accuracy: {(accuracy/len(valid_dataloaders)) * 100 :.3f}%")
                running_loss = 0
                model.train()
    return model
    

def save_checkpoint(arch, model, save_dir, optimizer, class_to_idx):
    filename = arch+'_checkpoint.pth'
    if save_dir is not None:
        filename = save_dir + '/'+filename
        
    checkpoint = {
    'arch': arch,
    'state_dict': model.state_dict(),
    'optimizer_dict': optimizer.state_dict(),
    'class_to_idx': class_to_idx,
    'classifier': model.classifier,
    }
    torch.save(checkpoint, filename) 
    
# Call to main function to run the program
if __name__ == "__main__":
    main()

   
  