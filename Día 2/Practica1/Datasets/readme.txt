
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Description of the files containing the datasets~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 

The two files sat-4-full.mat and sat-6-full.mat contain the full SAT-4 and SAT-6 datasets respectively.

The entire SAT-4 dataset consists of a labelled set of 400,000 training samples and 100,000 test samples and has a size of ~1.36GB. The SAT-6 dataset 
consists of a labelled set of 324,000 training samples and 81,000 test samples and has a size of ~1.12GB. 

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~File descriptions~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

sat-4-full.mat and sat-6-full.mat are MATLAB mat files that can be loaded into MATLAB using the standard ‘load’ function.

sat-4-full.mat contains the following variables:

train_x        --------------    28x28x4x400000 uint8  (containing 400000 training samples of 28x28 images each with 4 channels - R, G, B and NIR)
train_y        --------------    4x400000       double (containing 4x1 vectors having labels for the 400000 training samples)
test_x         --------------    28x28x4x100000 uint8  (containing 100000 test samples of 28x28 images each with 4 channels - R, G, B and NIR)
test_y         --------------    4x100000       double (containing 4x1 vectors having labels for the 100000 test samples) 
annotations    --------------    4x2            cell   (containing the class label annotations for the 4 classes of SAT-4)

sat-6-full.mat contains the following variables:

train_x        --------------    28x28x4x324000 uint8  (containing 324000 training samples of 28x28 images each with 4 channels - R, G, B and NIR)
train_y        --------------    6x324000       double (containing 6x1 vectors having labels for the 324000 training samples)
test_x         --------------    28x28x4x81000  uint8  (containing 81000 test samples of 28x28 images each with 4 channels - R, G, B and NIR)
test_y         --------------    6x81000        double (containing 6x1 vectors having labels for the 81000 test samples)
annotations    --------------    6x2            cell   (containing the class label annotations for the 6 classes of SAT-6)

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


 