# Auto-encoding-basedAnomaly  Detection  forIntrusionDetection   via   ClAssification (AIDA)


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

## Example (Optional)

```javascript
// code away!

let generateProject = project => {
  let code = [];
  for (let js = 0; js < project.length; js++) {
    code.push(js);
  }
};
```

