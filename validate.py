#import all imports and parameters from import_parameters.py

from import_parameters import *

from defs import load_checkpoint, build_classifier
from defs import validate_model





model, criterion = load_checkpoint()
model.to(device)
validate_accuracy_rate, validate_loss_rate = validate_model(model , criterion , dataloaders['validate'])

print('For the validation set, the accuracy rate is {:.3}'.format(validate_accuracy_rate))
print('For the validation set, the loss rate is {:.3}'.format(validate_loss_rate))