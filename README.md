# Auto-encoding-basedAnomaly  Detection  forIntrusionDetection   via   ClAssification (AIDA)

The repository contains code refered to the work:

_Giuseppina Andresini, Annalisa Appice, Nicola Di Mauro, Corrado Loglisci, Donato Malerba_

Exploiting the Auto-Encoder Residual Error for Intrusion Detection 


## Code requirements

The code relies on the following **python3.6+** libs.

Packages need are:
* [Tensorflow 1.13](https://www.tensorflow.org/) 
* [Keras 2.2.4](https://github.com/keras-team/keras) 
* [Matplotlib 3.0.3](https://matplotlib.org/)
* [Numpy 1.15.4](https://www.numpy.org/)
* [Scikit-learn](https://scikit-learn.org/stable/)

## Data
The 


## How to use
Repository contains script of different baseline:
* __C1+A1__ : script Autoencoder1+ Classification contains code to execute Autoecndoer and Classifcation cascade without error residual feature
* __AIDA__ : script AIDA contains entire step to execute the three AIDA phase: 
  * __Autoencoder1__ model is used for residual-error feature augmentation (Section III-B)
  * __Classification__ model is a neural netowrk with a final softmax layer which use data augmented with residual-error (Section     III-D)
  * __Autoencoder2__ is used for residual-error anomaly-based post-classification (Section III-D)
  
 Code contains models (autoencoder and classification) and datasets used for experiments in the work.
 
  

## Replicate the experiments

To replicate experiments reported in the work, you can use global variable and models and datasets stored in homonym folders.


```python
    N_CLASSES = 2
    PREPROCESSING1 = 0  #if set to 1 code execute preprocessing phase ( categorical to numeric, one-hot encode, standard scale) on original date
    PREPROCESSING2 = 0  #if set to 1 code execute preprocessing phase ( categorical to numeric, one-hot encode, standard scale) on data augmented
    LOAD_AUTOENCODER1 = 1 #if 1 load autoencoder1 from models folder
    LOAD_CLASSIFIER = 1  #if 1 load classifier  from models folder
    LOAD_MODEL = 1  #if 1 load autoencoder2 from models folder
```


