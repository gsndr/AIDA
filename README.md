# Auto-encoding-basedAnomaly  Detection  forIntrusionDetection   via   ClAssification (AIDA)

The repository contains code refered to the work:
*Giuseppina Andresini, Annalisa Appice, Nicola Di Mauro, Corrado Loglisci, Donato Malerba *\
Exploiting the Auto-Encoder Residual Error forIntrusion Detection \


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
  
 Code contains models (autoencoder and classification) and datasets used for experiments in 
  


## Variable(Optional)

```python
    N_CLASSES = 2
    PREPROCESSING1 = 0
    PREPROCESSING2 = 0
    LOAD_AUTOENCODER1 = 1
    LOAD_CLASSIFIER = 1
    LOAD_MODEL = 1
    VALIDATION_SPLIT = .1
```

