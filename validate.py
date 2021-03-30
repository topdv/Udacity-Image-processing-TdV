#import all imports and parameters from import_parameters.py

from import_parameters import *

parser = argparse.ArgumentParser(description = 'All arguments to be used in predicting')
parser.add_argument ('--checkpoint', default = 'checkpoint.pth',  help = 'Filename of checkpoint (default = checkpoint.pth)', metavar = '')
parser.add_argument ('--arch', metavar='ARCH', default='vgg16', help='Chose between two options; densenet161 or vgg16 (default is vgg16)')
parser.add_argument ('--device', default = 'GPU', help = 'Chose which device will be used to predict (default is GPU if available, otherwise CPU)')
parser.add_argument ('--train_dir', default = 'flowers/train',  help = 'Directory of training fotos, this needs to be mentioned', metavar = '')
parser.add_argument ('--image_path', default = "flowers/test/10/image_07090.jpg",  help = 'Image path of immage to be classified (default is flowers/test/10/image_07090.jpg)', metavar = '')
parser.add_argument ('--hidden_units', default = 512, type=int, help = 'Number of units in the first hidden layer (default is 512)')

args = parser.parse_args()


filepath = args.checkpoint
arch = args.arch
image_path = args.image_path
hidden_units = args.hidden_units


#defining if model should be run on cpu or GPU
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












print('setting parameters')
#Setting directories
data_dir = 'flowers'
train_dir = args.train_dir
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


print('define data transforms')
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

print('define what are the datasets')
# Load the datasets with ImageFolder
train_dir = args.train_dir

image_datasets = {
    'train': datasets.ImageFolder(train_dir, transform= data_transforms['train']),
    'validate': datasets.ImageFolder(valid_dir, transform = data_transforms['validate']),
    'test': datasets.ImageFolder(test_dir, transform=data_transforms['test'])
}




print('define dataloader')
# Define the dataloaders using the image datasets and the trainforms
dataloaders = {
    'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size = 64, shuffle = True),
    'validate': torch.utils.data.DataLoader(image_datasets['validate'], batch_size = 32),
    'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size = 32)
}





print('define model architecture')
## define model architecture
#Define model and classifier size dependant on chosen model
arch = args.arch
model = models.__dict__[args.arch](pretrained=True)
if arch == "vgg16":
    layers = [25088, hidden_units, 200, 102]

elif arch == 'densenet161':
    layers = [2208, hidden_units, 200, 102]

# model = models.vgg16(pretrained=True)

# don't compute gradients
for param in model.parameters():
    param.requires_grad = False




#defining build_classifier
print('define build_classifier')
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











#defining load_checkpoint
print('define load_checkpoint')
def load_checkpoint(filepath, arch):

    checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
    model = models.vgg16(pretrained=True)

    # Freeze the feature parameters
    for params in model.parameters():
        params.requires_grad = False

    #create new classifier
    classifier = build_classifier(layers)
    model.classifier = classifier


    criterion = nn.NLLLoss()

    optimizer = optim.Adam(model.classifier.parameters(), lr = 0.001)


    model.class_to_idx = checkpoint['class_to_idx']

    model.load_state_dict(checkpoint['model_state_dict'])

    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, criterion























print('define validate_model')
def validate_model(model , criterion , dataloader ):
    model.eval()
    model.cuda()
    sum_loss = 0
    sum_accuracy = 0

    for data in iter(dataloader):
        inputs, labels = data

        inputs = inputs.float().cuda()
        labels = labels.long().cuda()

        inputs = Variable(inputs)
        labels = Variable(labels)

        output = model.forward(inputs)
        loss = criterion(output, labels)
        sum_loss += loss
        ps = torch.exp(output).data

        equality = labels.data == ps.max(1)[1]
        sum_accuracy += equality.type_as(torch.FloatTensor()).mean()

    loss_rate = sum_loss / len(dataloader)
    accuracy_rate = sum_accuracy / len(dataloader)

    return accuracy_rate, loss_rate


model, criterion = load_checkpoint(filepath, model)
model.to(device)
validate_accuracy_rate, validate_loss_rate = validate_model(model , criterion , dataloaders['train'])

print('For the validation set, the accuracy rate is {:.3}'.format(validate_accuracy_rate))
print('For the validation set, the loss rate is {:.3}'.format(validate_loss_rate))
