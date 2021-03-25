from import_parameters import *
# from validate import data_transforms


from defs import load_checkpoint, build_classifier
from defs import imshow, process_image, predict, display

import argparse


parser = argparse.ArgumentParser(description = 'This function will predict the probability of a flower given a picture of this flower')

parser.add_argument ('--image_path', default = "flowers/test/10/image_07090.jpg",  help = 'Image path of immage to be classified', metavar = '')
parser.add_argument ('--checkpoint', default = "checkpoint.pth", help = 'Path to checkpoint to be used', metavar = '')

args = parser.parse_args()

# args = arg_parser_predict()

# image_path = args.Path

# print(args)


model, criterion = load_checkpoint(args.checkpoint)

# image_path = "flowers/test/10/image_07090.jpg"



# probs, labels = predict(image_path, model, topk=5)

probs, labels, classes, title = display(args.image_path, model)
class_names = [cat_to_name[item] for item in classes]



np.set_printoptions(precision=2)



print("="*80)
print(" "*35 + 'FLOWER PREDICTOR')
print("="*80)
print("Input label (or labels) =   {}".format(classes))
print("Probability confidence(s) = {}".format(probs))
print("Class(es) name(s) =            {}".format(class_names))
print("="*80)