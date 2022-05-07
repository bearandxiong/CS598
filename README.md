# CS598
Source codes are stored in ./code folder.



#### 1. Specification of dependencies

Required python module: tensorflow, keras, numpy, pandas, sklearn, h5py and scipy.
These are standard python modules and can be installed using pip install.

Also need keract, whose source code can be found here: https://github.com/philipperemy/keract


#### 2. Data pre-processing code
Code name: getRawData.py
Main function name: getRawData()
Usage: read in raw text MIMIC data and parse the text data into pandas, save in excel file.

Code name: preProcessing.py
Main function name: preProcessing()
Usage: convert the text file into sparse numerical representation, output model ready data.


#### 3. Training code
Code name: trainAutoEncoder.py
Usage: train auto encoder to extract condense information from text input.

Code name: loadTrainingData.py
Usage: load training data generator.

Code name: model.py
Main class: myModel
Usage: building the model infractructure

Code name: trainModel.py
Usage: training the main model, and generated synthetic text, save to an excel file.



#### 4. Evaluation code

Code name: evaluateModel.py
Usage: use the pre-trained model, calculate ppv, sens, f1, CIDEr, ES scores for various sampling approaches.

Code name: classification.py
Usage: train the cnn model as a classifier, to classify ICD code based on chief complaints and other variables (gender, age etc.)

#### 5. Pre-trained models

Training a model from scratch can be time-consuming and expensive, therefore I have the pre-trained model under ./model.
1) auto_encoder.hdf5: encoder model for encoding text inputs.
2) trainedModel.hdf5: trained main model.
3) trainedClassifier.hdf5: classifier based on rnn.

