print('hello world!')

#import all imports and parameters from import_parameters.py

from import_parameters import *
from defs import build_classifier, train_model, save_model





# lr = arg_parser_train().Lr
lr = Lr
# num_epochs = arg_parser().num_epochs
num_epochs = 3

#Define classifier
classifier = build_classifier(layers)
model.classifier = classifier

#Define criterion
criterion = nn.NLLLoss()

#Define optimizer
optimizer = optim.Adam(model.classifier.parameters(), lr = lr)





#Execute all modules to train the classifier

train_model(model, criterion, optimizer, num_epochs)
save_model(num_epochs, model, optimizer)


