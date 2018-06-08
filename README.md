# Digit Recognizer

The data files train.csv and test.csv contain gray-scale images of hand-drawn digits, from zero through nine.

Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255, inclusive.

The training data set, (train.csv), has 785 columns. The first column, called "label", is the digit that was drawn by the user. The rest of the columns contain the pixel-values of the associated image.

Each pixel column in the training set has a name like pixelx, where x is an integer between 0 and 783, inclusive. To locate this pixel on the image, suppose that we have decomposed x as x = i * 28 + j, where i and j are integers between 0 and 27, inclusive. Then pixelx is located on row i and column j of a 28 x 28 matrix, (indexing by zero).

For example, pixel31 indicates the pixel that is in the fourth column from the left, and the second row from the top, as in the ascii-diagram below.

Visually, if we omit the "pixel" prefix, the pixels make up the image like this:

000 001 002 003 ... 026 027
028 029 030 031 ... 054 055
056 057 058 059 ... 082 083
 |   |   |   |  ...  |   |
728 729 730 731 ... 754 755
756 757 758 759 ... 782 783 
The test data set, (test.csv), is the same as the training set, except that it does not contain the "label" column.

Your submission file should be in the following format: For each of the 28000 images in the test set, output a single line containing the ImageId and the digit you predict. For example, if you predict that the first image is of a 3, the second image is of a 7, and the third image is of a 8, then your submission file would look like:

ImageId,Label
1,3
2,7
3,8 
(27997 more lines)
The evaluation metric for this contest is the categorization accuracy, or the proportion of test images that are correctly classified. For example, a categorization accuracy of 0.97 indicates that you have correctly classified all but 3% of the images.

## My Approach ::

1. Using Simple Classifier (SVM):
I started with using a simple SVM on the dataset and found that the accuracy came out to be 55% then I normalzed the data by subtracting by mean and dividing by the max value(255 here) and accuracy went upto 88%.

2. Using Logisitic Regression :
As the labels were multiple in number so I used softmax at last to find the probabilities and saw that the model was doing fairly well.
Logistic Regression

    Linear regression is not good at classification.
    We use logistic regression for classification.
    linear regression + logistic function(softmax) = logistic regression
    Check my deep learning tutorial. There is detailed explanation of logistic regression.
    Steps of Logistic Regression
        Import Libraries
        Prepare Dataset
            We use MNIST dataset.
            There are 28*28 images and 10 labels from 0 to 9
            Data is not normalized so we divide each image to 255 that is basic normalization for images.
            In order to split data, w use train_test_split method from sklearn library
            Size of train data is 80% and size of test data is 20%.
            Create feature and target tensors. At the next parts we create variable from these tensors. As you remember we need to define variable for accumulation of gradients.
            batch_size = batch size means is that for example we have data and it includes 1000 sample. We can train 1000 sample in a same time or we can divide it 10 groups which include 100 sample and train 10 groups in order. Batch size is the group size. For example, I choose batch_size = 100, that means in order to train all data only once we have 336 groups. We train each groups(336) that have batch_size(quota) 100. Finally we train 33600 sample one time.
            epoch: 1 epoch means training all samples one time.
            In our example: we have 33600 sample to train and we decide our batch_size is 100. Also we decide epoch is 29(accuracy achieves almost highest value when epoch is 29). Data is trained 29 times. Question is that how many iteration do I need? Lets calculate:
                training data 1 times = training 33600 sample (because data includes 33600 sample)
                But we split our data 336 groups(group_size = batch_size = 100) our data
                Therefore, 1 epoch(training data only once) takes 336 iteration
                We have 29 epoch, so total iterarion is 9744(that is almost 10000 which I used)
            TensorDataset(): Data set wrapping tensors. Each sample is retrieved by indexing tensors along the first dimension.
            DataLoader(): It combines dataset and sampler. It also provides multi process iterators over the dataset.
            Visualize one of the images in dataset
        Create Logistic Regression Model
            Same with linear regression.
            However as you expect, there should be logistic function in model right?
            In pytorch, logistic function is in the loss function where we will use at next parts.
        Instantiate Model Class
            input_dim = 2828 # size of image pxpx
            output_dim = 10 # labels 0,1,2,3,4,5,6,7,8,9
            create model
        Instantiate Loss Class
            Cross entropy loss
            It calculates loss that is not surprise :)
            It also has softmax(logistic function) in it.
        Instantiate Optimizer Class
            SGD Optimizer
        Traning the Model
        Prediction


3. Using Deep Learning :
a. Using ANN :


a. Using CNN+Keras :

    1. Introduction
    2. Data preparation
        2.1 Load data
        2.2 Check for null and missing values
        2.3 Normalization 
        2.4 Reshape 
        2.5 Label encoding (One hot encoding)
        2.6 Split training and valdiation set
    3. CNN
        3.1 Define the model
        3.2 Set the optimizer and annealer()
        3.3 Data augmentation
    4. Evaluate the model
        4.1 Training and validation curves
        4.2 Confusion matrix
    5. Prediction and submition
        5.1 Predict and Submit results

 Steps of CNN:

    Import Libraries
    Prepare Dataset
        Totally same with previous parts.
        We use same dataset so we only need train_loader and test_loader.
    Convolutional layer:
        Create feature maps with filters(kernels).
        Padding: After applying filter, dimensions of original image decreases. However, we want to preserve as much as information about the original image. We can apply padding to increase dimension of feature map after convolutional layer.
        We use 2 convolutional layer.
        Number of feature map is out_channels = 16
        Filter(kernel) size is 5*5
    Pooling layer:
        Prepares a condensed feature map from output of convolutional layer(feature map)
        2 pooling layer that we will use max pooling.
        Pooling size is 2*2
    Flattening: Flats the features map
    Fully Connected Layer:
        Artificial Neural Network that we learnt at previous part.
        Or it can be only linear like logistic regression but at the end there is always softmax function.
        We will not use activation function in fully connected layer.
        You can think that our fully connected layer is logistic regression.
        We combine convolutional part and logistic regression to create our CNN model.
    Instantiate Model Class
        create model
    Instantiate Loss Class
        Cross entropy loss
        It also has softmax(logistic function) in it.
    Instantiate Optimizer Class
        SGD Optimizer
    Traning the Model
    Prediction

As a result, as you can see from plot, while loss decreasing, accuracy is increasing and our model is learning(training).
Thanks to convolutional layer, model learnt better and accuracy(almost 98%) is better than accuracy of ANN. Actually while tuning hyperparameters, increase in iteration and expanding convolutional neural network can increase accuracy but it takes too much running time that we do not want at kaggle. 

I used the Keras Sequential API, where you have just to add one layer at a time, starting from the input.

The first is the convolutional (Conv2D) layer. It is like a set of learnable filters. I choosed to set 32 filters for the two firsts conv2D layers and 64 filters for the two last ones. Each filter transforms a part of the image (defined by the kernel size) using the kernel filter. The kernel filter matrix is applied on the whole image. Filters can be seen as a transformation of the image.

The CNN can isolate features that are useful everywhere from these transformed images (feature maps).

The second important layer in CNN is the pooling (MaxPool2D) layer. This layer simply acts as a downsampling filter. It looks at the 2 neighboring pixels and picks the maximal value. These are used to reduce computational cost, and to some extent also reduce overfitting. We have to choose the pooling size (i.e the area size pooled each time) more the pooling dimension is high, more the downsampling is important.

Combining convolutional and pooling layers, CNN are able to combine local features and learn more global features of the image.

Dropout is a regularization method, where a proportion of nodes in the layer are randomly ignored (setting their wieghts to zero) for each training sample. This drops randomly a propotion of the network and forces the network to learn features in a distributed way. This technique also improves generalization and reduces the overfitting.

'relu' is the rectifier (activation function max(0,x). The rectifier activation function is used to add non linearity to the network.

The Flatten layer is use to convert the final feature maps into a one single 1D vector. This flattening step is needed so that you can make use of fully connected layers after some convolutional/maxpool layers. It combines all the found local features of the previous convolutional layers.

In the end i used the features in two fully-connected (Dense) layers which is just artificial an neural networks (ANN) classifier. In the last layer(Dense(10,activation="softmax")) the net outputs distribution of probability of each class.

3. Using Dimensionality Reduction :

    Principal Component Analysis ( PCA ) - Unsupervised, linear method

    Linear Discriminant Analysis (LDA) - Supervised, linear method

    t-distributed Stochastic Neighbour Embedding (t-SNE) - Nonlinear, probabilistic method

PCA -> min error and max variance

4. Using RNN:
Recurrent Neural Network (RNN)

    RNN is essentially repeating ANN but information get pass through from previous non-linear activation function output.
    Steps of RNN:
        Import Libraries
        Prepare Dataset
        Create RNN Model
            hidden layer dimension is 100
            number of hidden layer is 1
        Instantiate Model Class
        Instantiate Loss Class
            Cross entropy loss
            It also has softmax(logistic function) in it.
        Instantiate Optimizer Class
            SGD Optimizer
        Traning the Model
        Prediction


