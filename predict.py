from import_parameters import *
# from validate import data_transforms





parser = argparse.ArgumentParser(description = 'All arguments to be used in predicting')
parser.add_argument ('--checkpoint', default = 'checkpoint.pth',  help = 'filename of checkpoint', metavar = '')
parser.add_argument ('--arch', metavar='ARCH', default='vgg16', help='Chose between two options; densenet161 or vgg16 (default)')
parser.add_argument ('--image_path', default = "flowers/test/10/image_07090.jpg",  help = 'Image path of immage to be classified', metavar = '')
parser.add_argument ('--topk', default = 5, type = int, help = 'The number of highest probabilities for the given prediction (default = 5)')
parser.add_argument ('--device', default = 'GPU', help = 'Chose which device will be used to predict (default is GPU if available, otherwise CPU)')
parser.add_argument ('--mapping', default = 'cat_to_name.json', help = 'Chose the mapping file for labeling the flower probabilities (default is cat_to_name.json)')
parser.add_argument ('--hidden_units', default = 512, type=int, help = 'Number of units in the first hidden layer (default is 512)')

args = parser.parse_args()

filepath = args.checkpoint
arch = args.arch
image_path = args.image_path
topk = args.topk
mapping = args.mapping
hidden_units = args.hidden_units




#Loading JSON file with name mapping
with open(mapping, 'r') as f:
    cat_to_name = json.load(f)




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










print('define model architecture')
## define model architecture
#Define model and classifier size dependant on chosen model
arch = args.arch
model = models.__dict__[args.arch](pretrained=True)
if arch == "vgg16":
    layers = [25088, hidden_units, 200, 102]

elif arch == 'densenet161':
    layers = [2208, hidden_units, 200, 102]


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
    model = models.__dict__[arch](pretrained=True)

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


#defining build_classifier

#defining imshow --> Not used

#defining process_image
print ('define process_image')
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    #Open image
    im = Image.open(image)

    #Resize picture
    # Get dimensions.
    width, height = im.size

    # Find shorter size and create settings to crop shortest side to 256
    if width < height:
        size=[256, 999]
    else:
        size=[999, 256]

    im.thumbnail(size)

    #Crop picture
    crop_size = 244
    center = width/4, height/4
    crop_dimensions = (center[0]-(crop_size/2), center[1]-(crop_size/2), center[0]+(crop_size/2), center[1]+(crop_size/2))
    im = im.crop(crop_dimensions)

    #Adjust for number of color chanels
    numpy_img = np.array(im)/255

    # Normalize each color channel
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    numpy_img = (numpy_img-mean)/std

    # Set the color to the first channel
    numpy_img = numpy_img.transpose(2, 0, 1)

    return numpy_img

#defining predict
print('define predict')
def predict(image_path, model, topk=topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.to(device)
    model.eval();

    # Define model class to IDx
#     model.class_to_idx = image_datasets['train'].class_to_idx


    #load image in numpy array
    numpy_immage = process_image(image_path)

    #Convert numpy array to tensor
    tensor_image = np.expand_dims(numpy_immage, axis=0)


    #Convert tensor to torch image
    torch_image = torch.from_numpy(tensor_image).type(torch.FloatTensor)
    torch_image = torch_image.to(device)

    output = model(torch_image)
    torch.exp_(output)


    probs, labels = output.topk(topk)

    probs = np.array(probs.data)[0]
    labels = np.array(labels)[0]

    return probs, labels

#defining display
print ('define display')
model = args.arch

def display (image_path, model = model):

    probs, labels = predict(image_path, model)
    processed_image = process_image(image_path)

    mapping = {val: key for key, val in
                model.class_to_idx.items()
                }

    classes = [mapping[label] for label in labels]

#     label_map = {v: k for k, v in model.class_to_idx.items()}

#     classes = [cat_to_name[label_map[label]] for label in labels]


    title = cat_to_name[image_path.split('/')[-2]]

    return probs, labels, classes, title






# args = arg_parser_predict()

# image_path = args.Path

# print(args)


model, criterion = load_checkpoint(filepath, model)

# image_path = "flowers/test/10/image_07090.jpg"



# probs, labels = predict(image_path, model, topk=5)

probs, labels, classes, title = display(image_path, model)
class_names = [cat_to_name[item] for item in classes]



np.set_printoptions(precision=2)



print("="*80)
print(" "*35 + 'FLOWER PREDICTOR')
print("="*80)
print("Input label (or labels) =   {}".format(classes))
print("Probability confidence(s) = {}".format(probs))
print("Class(es) name(s) =            {}".format(class_names))
print("="*80)
