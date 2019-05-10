# Auto-encoding-basedAnomaly  Detection  for IntrusionDetection   via   ClAssification (AIDA)

The repository contains code refered to the work:

_Giuseppina Andresini, Annalisa Appice, Nicola Di Mauro, Corrado Loglisci, Donato Malerba_

Exploiting the Auto-Encoder Residual Error for Intrusion Detection 


## Code requirements

The code relies on the following **python3.6+** libs.

Packages need are:
* [Tensorflow 1.13](https://www.tensorflow.org/) 
* [Keras 2.2.4](https://github.com/keras-team/keras) 
* [Matplotlib 3.0.3](https://matplotlib.org/)
* [Pandas 0.24.2](https://pandas.pydata.org/)
* [Numpy 1.15.4](https://www.numpy.org/)
* [Scikit-learn](https://scikit-learn.org/stable/)

## Data
The dataset used for experiments is accessible from [__NSL-KDD__](https://www.unb.ca/cic/datasets/nsl.html). Original dataset is transformed in a binary classification: "_attack_, _normal_" (_oneCls files) and then the  feature  selection  stage  is  performed  by  retain  the  10top-ranked  features  according  to  __Information  Gain(IG)__ .

After applying the one-hot encoder mapping to transform the selectedsymbolic features in quantitative ones, an input feature spacewith 89 quantitative features is finally constructed (_Numeric files).

This input features  space  is  expanded  with  the  addition  of  the  residual error feature that is engineered using the auto-encoder trainedon the non-attacking training data. (_mse_Numeric files)

## How to use
Repository contains scripts of different baseline:
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


