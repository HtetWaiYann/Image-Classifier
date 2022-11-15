from torch import nn
from torchvision import models

def load_model_and_classifier(arch, hidden_units):

    if arch == 'alexnet':
        model = models.alexnet(pretrained=True)
    else:
        model = models.densenet121(pretrained=True)
    # Freeze the model params
    for param in model.parameters():
        param.requires_grad = False
        
    # Create the classifier for the model
    model.classifier = create_classifier(arch, hidden_units)

    return model


def create_classifier(arch, hidden_units):
    # Classifier for alexnet model
    if arch == 'alexnet':
        classifier = nn.Sequential(
            nn.Linear(9216, hidden_units),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_units, 102),
            nn.LogSoftmax(dim=1)
        )
    # Classifier for densenet model
    else:
        classifier = nn.Sequential(
            nn.Linear(1024, hidden_units),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_units, 102),
            nn.LogSoftmax(dim=1)
        )
    return classifier
    