# Comp4901J Deep Learning In Computer Vision - Assignment1

The course website: https://course.cse.ust.hk/comp4901j/

Image Classification, kNN, SVM, Softmax, Neural Network

In this assignment you will practice putting together a simple image classification pipeline, based on the k-Nearest Neighbor or the SVM/Softmax classifier. The goals of this assignment are as follows:

understand the basic Image Classification pipeline and the data-driven approach (train/predict stages)
understand the train/val/test splits and the use of validation data for hyperparameter tuning.
develop proficiency in writing efficient vectorized code with numpy
implement and apply a k-Nearest Neighbor (kNN) classifier
implement and apply a Multiclass Support Vector Machine (SVM) classifier
implement and apply a Softmax classifier
implement and apply a Two layer neural network classifier
understand the differences and tradeoffs between these classifiers
get a basic understanding of performance improvements from using higher-level representations than raw pixels (e.g. color histograms, Histogram of Gradient (HOG) features)

Download data: Once you have the starter code, you will need to download the CIFAR-10 dataset. Run the following from the assignment1 directory:

cd cs231n/datasets folder and click get_datasets.py to run the Python code to download the data.
Submitting your work:

Whether you work on the assignment locally or using Terminal, once you are done working run the Python code collectSubmission.py; this will produce a file called assignment1.7z. Upload this file to CASS. You can find CASS instruction of uploading your submission here.

Q1: k-Nearest Neighbor classifier (20 points)

The IPython Notebook knn.ipynb will walk you through implementing the kNN classifier.

Q2: Training a Support Vector Machine (25 points)

The IPython Notebook svm.ipynb will walk you through implementing the SVM classifier.

Q3: Implement a Softmax classifier (20 points)

The IPython Notebook softmax.ipynb will walk you through implementing the Softmax classifier.

Q4: Two-Layer Neural Network (25 points)

The IPython Notebook two_layer_net.ipynb will walk you through the implementation of a two-layer neural network classifier.

Q5: Higher Level Representations: Image Features (10 points)

The IPython Notebook features.ipynb will walk you through this exercise, in which you will examine the improvements gained by using higher-level representations as opposed to using raw pixel values.

Q6: Cool Bonus: Do something extra! (+10 points)

Implement, investigate or analyze something extra surrounding the topics in this assignment, and using the code you developed. For example, is there some other interesting question we could have asked? Is there any insightful visualization you can plot? Or anything fun to look at? Or maybe you can experiment with a spin on the loss function? If you try out something cool we'll give you up to 10 extra points and may feature your results in the lecture.
