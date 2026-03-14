# trainEasy
  
**trainEasy**, is a user-friendly Python wrapper for building fully-connected deep Artificial Neural Network (ANN) models
 with an arbitrary structure. In the ./examples/ dir, we provide a case where a 16-layer and 2.8 billion parameter ANN
 is built using a single line of code. The user simply provides the network structure in a list, and the model is automatically built. 
 
 For instance, to build a model having 6 layers with 10, 400, 200, 100, 50, 5 nodes in each layer,
 the user simply creates an object of the FcNetwork class using, 
 
 model=FcNetwork([10, 400, 200, 100, 50, 5]). 

 The list may also contain the following types of layers: 

 Dropout: {'dp': 0.5}
 
 Batch Normalisastion: 'bn'
 
 Normalisation: 'n' 

**Author: Z. Nikolaou (2026).** 

**Contact: ZachariasMNic@gmail.com**

Installation: 
-------------

1. Clone the repo to your local machine.  

2. Set environment variables: 
   
   *source setenv.sh*
    
To use:
-------

Have a look at the *./examples/* directory.
