#### All functions defined below

from import_parameters import *

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


def train_model(model, criterion, optimizer, num_epochs):

   
    print_every = 5
    steps = 0
    model.cuda()
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


def load_checkpoint(filepath, torch = torch):
    checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
    
    # TODO: Build and train your network
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


def save_model(num_epochs, model, optimizer):
    model.class_to_idx = image_datasets['train'].class_to_idx
    checkpoint = {
              'num_epochs': num_epochs,
              'class_to_idx': model.class_to_idx,
              'layers': [25088, 1024, 200, 102],
              'optimizer_state_dict': optimizer.state_dict(), 
              'model_state_dict': model.state_dict()}
    
    torch.save(checkpoint, "checkpoint.pth")
    return

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

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
#     # PyTorch tensors assume the color channel is the first dimension
#     # but matplotlib assumes is the third dimension
#     image = image.transpose(1, 2, 0)
    
#     # Undo preprocessing
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     image = std * image + mean
    
#     # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
#     image = np.clip(image, 0, 1)
    
#     ax.imshow(image)
    
#     return ax


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.to("cpu")
    model.eval();
    
    # Define model class to IDx
    model.class_to_idx = image_datasets['train'].class_to_idx

    
    #load image in numpy array
    numpy_immage = process_image(image_path)

    
    #Convert numpy array to tensor
    tensor_image = np.expand_dims(numpy_immage, axis=0)
  

    #Convert tensor to torch image
    torch_image = torch.from_numpy(tensor_image).type(torch.FloatTensor)
 
    
    output = model.forward(torch_image)
    torch.exp_(output)
       
       
    probs, labels = output.topk(topk)
    
    probs = np.array(probs.data)[0]
    labels = np.array(labels)[0]

    return probs, labels


# TODO: Display an image along with the top 5 classes
# def display (image_path, model = model):
#     probs, labels = predict(image_path, model)
#     processed_image = process_image(image_path)
    
#     label_map = {v: k for k, v in model.class_to_idx.items()}
    
#     classes = [cat_to_name[label_map[l]] for l in labels]
    
#     title = cat_to_name[image_path.split('/')[-2]]
#     f, (ax1, ax2) = plt.subplots(2, 1, figsize = (6,6))
#     plt.tight_layout()

#     imshow(processed_image, ax=ax1, title=title)

#     ax1.set_xticks([])
#     ax1.set_yticks([])

#     class_ticks = np.arange(len(classes))
#     ax2.barh(class_ticks, probs)
#     ax2.invert_yaxis()
#     ax2.set_yticks(class_ticks)
#     ax2.set_yticklabels(classes)




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



        
    