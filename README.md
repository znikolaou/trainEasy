# trainEasy
  
**trainEasy**, is a user friendly Python wrapper for constructing deep Artificial Neural Network (ANN) models
 with an arbitraty structure. The user simply provides the structure of the network in a list, and the model is automatically built, and 
 trained. For instance, to build a model having 6 layers with 10, 400, 200, 100, 50, 5 nodes in each layer,
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
