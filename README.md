# Auto-encoding-basedAnomaly  Detection  forIntrusionDetection   via   ClAssification (AIDA)

The repository contains code refered to the work:

_Giuseppina Andresini, Annalisa Appice, Nicola Di Mauro, Corrado Loglisci, Donato Malerba_

Exploiting the Auto-Encoder Residual Error forIntrusion Detection 


## Code requirements

The code relies on the following **python3.6+** libs.

Packages need are:
* [Tensorflow 1.13](https://www.tensorflow.org/) 
* [Keras 2.2.4](https://github.com/keras-team/keras) 
* [Matplotlib 3.0.3](https://matplotlib.org/)
* [Numpy 1.15.4](https://www.numpy.org/)
* [Scikit-learn](https://scikit-learn.org/stable/)


## How to use
Repository contains script of different baseline:
* C1+A1 : script Autoencoder1+ Classification contains code to execute Autoecndoer and Classifcation cascade without error residual feature
* AIDA : script AIDA contains entire step to execute the three AIDA phase: 
  * Autoencoder1 model is used for residual-error feature augmentation (Section III-B)
  * Classification model is a neural netowrk with a final softmax layer which use data augmented with residual-error (Section     III-D)
  * Autoencoder2 is used for residual-error anomaly-based post-classification (Section III-D)
  
 Code contains models (autoencoder and classification) and datasets used for experiments in the work.
  

## Variables to set to replicate experiments

To replicate experiments reported in the work, you can use global variable and models and datasets stored in homonym folders.


```python
    N_CLASSES = 2
    PREPROCESSING1 = 0  #if set to 1 code execute preprocessing phase ( categorical to numeric, one-hot encode, standard    scale) on original date
    PREPROCESSING2 = 0  #if set to 1 code execute preprocessing phase ( categorical to numeric, one-hot encode, standard  scale) on data augmented
    LOAD_AUTOENCODER1 = 1 #if 1 load autoencoder1 from models folder
    LOAD_CLASSIFIER = 1  #if 1 load classifier  from models folder
    LOAD_MODEL = 1  #if 1 load autoencoder2 from models folder
```


