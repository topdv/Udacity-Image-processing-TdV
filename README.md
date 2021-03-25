# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.


The image classifier is setup in several python scripts.

1. import_parameters.py to load all parameters for training or using in image recognition. This import file also contains the parser to be used in training and predicting.
2. import definitions using defs.py. In this script most of the functions are defined.
3. train.py used for training the classifier. This also includes the saving function to save the trained classifier. Optionally the training directory can be chosen as an argument using --train_dir
4. validate.py to do a validation withimages in the validate folder 
5. predit.py to make predictions with trained classifier. 
	By default the picture to be classified is 'flowers/test/10/image_07090.jpg', this can be changed with argument --image_path. 
	By default the model to be used it 'checkpoint.pth', optional this can be changed using the argument --checkpoint
