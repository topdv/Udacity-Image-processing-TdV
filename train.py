#import all imports and parameters from import_parameters.py

from import_parameters import *
import argparse


print('TRAIN.PY: define parser arguments for training')
parser = argparse.ArgumentParser(description = 'All arguments to be used in training')
parser.add_argument ('train_dir', default = 'flowers/train',  help = 'Directory of training fotos, this needs to be mentioned', metavar = '')
parser.add_argument ('--arch', metavar='ARCH', default='vgg16', help='Chose between two options; densenet161 or vgg16 (default)')
parser.add_argument ('--learningrate', default = 0.001, type= float, help = 'What learning rate should be used when training (default = 0.03)')
parser.add_argument ('--epochs', default = 3, type=int , help = 'Number of epochs (default = 1)')
parser.add_argument ('--device', default = 'GPU', help = 'Chose which device will be used to predict (default is GPU if available, otherwise CPU)')
parser.add_argument ('--hidden_units', default = 512, type=int, help = 'Number of units in the first hidden layer (default is 512)')


args = parser.parse_args()
hidden_units = args.hidden_units

#defining if model should be run on cpu or GPU
print('TRAIN.PY: defining device to run on')
if args.device == 'cpu':
    device = 'cpu'
    print('Device is set to cpu')
elif args.device == 'GPU':
    if (torch.cuda.is_available()):
        device = 'cuda'
        print('GPU device is available and will be set to GPU')

    else:
        device = 'cpu'
        print ('Device could not be set to GPU, therefore device is cpu')



print('TRAIN.PY: define model architecture')
## define model architecture
#Define model and classifier size dependant on chosen model
arch = args.arch
model = models.__dict__[arch](pretrained=True)
if arch == "vgg16":
    layers = [25088, hidden_units, 200, 102]

elif arch == 'densenet161':
    layers = [2208, hidden_units, 200, 102]

# model = models.vgg16(pretrained=True)

# don't compute gradients
for param in model.parameters():
    param.requires_grad = False




#defining build_classifier
print('TRAIN.PY: define build_classifier')
def build_classifier(layers):


    classifier  =    nn.Sequential(
                        nn.Linear(layers[0], layers[1]),
                        nn.ReLU(),
                        nn.Dropout(0.5), #50 % probability
                        nn.Linear(layers[1], layers[2]),
                        torch.nn.ReLU(),
                        torch.nn.Dropout(0.2), #20% probability
                        nn.Linear(layers[2], layers[3]),
                        nn.LogSoftmax(dim=1))

    return classifier





print('TRAIN.PY: Setting directories')
#Setting directories
data_dir = 'flowers'
train_dir = args.train_dir
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


print('TRAIN.PY: define data transforms')
data_transforms = {
    'train': transforms.Compose([
                                transforms.RandomRotation(random_rotation),
                                transforms.RandomResizedCrop(random_resize),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(network_means, network_stds)   ]),

    'validate' : transforms.Compose([
                                transforms.Resize(resize),
                                transforms.CenterCrop(center_crop),
                                transforms.ToTensor(),
                                transforms.Normalize(network_means, network_stds)   ]),

    'test' : transforms.Compose([
                                transforms.Resize(resize),
                                transforms.CenterCrop(center_crop),
                                transforms.ToTensor(),
                                transforms.Normalize(network_means, network_stds)  ])
}

print('TRAIN.PY: define what are the datasets')
# Load the datasets with ImageFolder
train_dir = args.train_dir

image_datasets = {
    'train': datasets.ImageFolder(train_dir, transform= data_transforms['train']),
    'validate': datasets.ImageFolder(valid_dir, transform = data_transforms['validate']),
    'test': datasets.ImageFolder(test_dir, transform=data_transforms['test'])
}




print('TRAIN.PY: define dataloader')
# Define the dataloaders using the image datasets and the trainforms
dataloaders = {
    'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size = 64, shuffle = True),
    'validate': torch.utils.data.DataLoader(image_datasets['validate'], batch_size = 32),
    'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size = 32)
}



#defining train_model
print('TRAIN.PY: define train_model')
def train_model(model, criterion, optimizer, num_epochs, train_dir = train_dir):

    print_every = 5
    steps = 0
    model.to(device)
    start_time_0 = time.localtime()
    start_time_1 = time.time()

    print('training commences')
    print(time.strftime("%H:%M:%S",start_time_0))


    for epoch in range (num_epochs):
        training_loss = 0
        training_accuracy = 0
        model.train()


        for inputs, labels in dataloaders['train']:
            steps += 1
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            # Forward and backward pass
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            training_loss += loss.item()
            ps_train = torch.exp(outputs).data
            equality_train = (labels.data == ps_train.max(1)[1])
            training_accuracy += equality_train.type_as(torch.FloatTensor()).mean()


            # Iteration to do the cross validation with the validation set
            if steps % print_every == 0:
                model.eval                                #Put in evaluation mode for validation
                validate_loss = 0
                validate_accuracy = 0

                for inputs_validate, labels_validate in dataloaders['validate']:
                    inputs_validate, labels_validate = inputs_validate.to(device), labels_validate.to(device)
                    with torch.no_grad():                 #Not needed as we will not backpropagate
                        outputs = model.forward(inputs_validate)
                        validate_loss = criterion(outputs,labels_validate)
                        ps = torch.exp(outputs).data
                        equality = (labels_validate.data == ps.max(1)[1])
                        validate_accuracy += equality.type_as(torch.FloatTensor()).mean()
                model.train()                            #Put model back in train mode

                validate_loss = validate_loss / len(dataloaders['validate'])
                validate_accuracy = validate_accuracy /len(dataloaders['validate'])


                #Print results for training and validation steps
                print("Epoch: {}/{}... ".format(epoch+1, num_epochs),
                      "Tr Loss: {:.2f}".format(training_loss/print_every),
                      "Val Loss {:.2f}".format(validate_loss),
                      "Tr Accuracy {:.2f}".format(training_accuracy/print_every),
                      "Val Accuracy: {:.2f}".format(validate_accuracy),
                      "Time spend on training (seconds): {:4.1f}".format(time.time() -start_time_1))
                training_loss = 0

    # Print results for the total time needed to train the model
    end_time_0 = time.localtime()
    end_time_1 = time.time()
    print('training started at {} and completed at {}, total training time was {}'.format(time.strftime("%H:%M:%S",start_time_0), time.strftime("%H:%M:%S"), end_time_0, (end_time_1 - start_time_1)))

#defining save_model
print('TRAIN.PY: define save_model')
def save_model(num_epochs, model, optimizer):
    model.class_to_idx = image_datasets['train'].class_to_idx
    checkpoint = {
              'num_epochs': num_epochs,
              'class_to_idx': model.class_to_idx,
              'layers': layers,
              'optimizer_state_dict': optimizer.state_dict(),
              'model_state_dict': model.state_dict()}

    torch.save(checkpoint, "checkpoint.pth")
    return













if __name__ == "__main__":
    #Execute all modules to train the classifier
    #extracting arguments from command line
    lr = args.learningrate
    num_epochs = args.epochs

    #Define classifier
    classifier = build_classifier(layers)
    model.classifier = classifier

    #Define criterion
    criterion = nn.NLLLoss()

    #Define optimizer
    optimizer = optim.Adam(model.classifier.parameters(), lr = lr)

    print('Train model starting')
    train_model(model, criterion, optimizer, num_epochs)
    print('save model starting')
    save_model(num_epochs, model, optimizer)
