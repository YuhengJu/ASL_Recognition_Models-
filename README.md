**Motivation**

According to the National Institute on Deafness and Other Communication Disorders, over 37 million
adults age 18+ in the U.S. report some type of trouble hearing. This represents a significant yet
underserved population. The sign-language economy, valued between $3 billion and $10 billion,
remains largely untapped, with many businesses only beginning to recognize its potential. However, the
lack of efficient technological solutions keeps ASL translation heavily reliant on human labor, limiting
scalability and accessibility.
To address these challenges and unlock the full potential of this market, a CNN-based ASL
recognition system is essential to bridge communication gaps and drive significant economic value.

**Exploratory Data Analysis**

Our data consists of 2,515 ASL hand signs for numbers 0-9 and letters A-Z. To store our images for
visualization, we created a data frame with categories 0-36 to encode these classes. Using this data
frame, we can display some images and their corresponding class.

In the corresponding plot, we display 100 out of the total 2515 images. All of the images are cropped
around the same size. This is beneficial for these cropped images to reduce noisy parts of the original
image by focusing only on the relevant features. However, if other images are introduced that are not
cropped, there may be some errors in our predictions. Another advantage of these images is that the ASL
hand signs have different orientations. This helps the model become more adaptable for the hand signs
in which it is trained, which are not always the same orientation.
To see if there is any imbalance in our dataset, we displayed the number of images in each class. 
Across the classes, we see 70 images for each class except for the “t” class, which has 65 images. This
is, thankfully, not a big difference in count as compared to the other classes. Thus, we don’t need to
apply any under sampling or oversampling to make the classes balanced.


**Data Preparation**

The data was split into three parts: training (80%), validation (10%), and testing (10%). The training,
validation, and testing data contained 2012, 251, and 252 images, respectively. The images were resized
to 200x200 to strike a balance between preserving the details of the image and keeping computations
manageable. All images were converted to tensors using transforms. ToTensor(), scaling pixel values to
the [0,1] range for consistent gradient updates during training. The image channel is 3 to represent RGB,
batch size is selected as 32 for our data loaders, and the number of classes is 36. Additionally, some of
the images were mislabeled with some images for “4” be actual “5” and vice versa. Thus, we made sure
all the images were in their appropriate class folders.

**Model Development**

We first implemented a simple linear model with a single fully connected layer. This model takes
flattened input and outputs logits for each class. We used callbacks like early stopping and a learning
rate scheduler to improve training efficiency.
Next, we created a custom CNN with three convolutional blocks, followed by fully connected layers.
Each convolutional block includes:

● Two 3x3 convolutional layers with ReLU activation to capture spatial features.

● Max-pooling layers with a kernel size of 2 to reduce spatial dimensions.

● Dropout with increasing rates (0.2, 0.3, 0.3) to prevent overfitting.


The number of filters increases progressively across blocks (32, 64, 128), reflecting hierarchical feature
extraction.

The fully connected layers consist of:

● 512 neurons in the first layer, followed by 128 neurons in the second.

● Dropout (0.25) applied between layers to avoid overfitting.

● The final layer has 36 neurons, corresponding to the number of output classes.


Training and Optimization:

● We used the Adam optimizer with an initial learning rate of 0.001 and a ReduceLROnPlateau
scheduler to dynamically adjust the learning rate.

● Categorical cross-entropy was used as the loss function for multi-class classification.

● Early stopping halted training if validation loss showed no improvement for five epochs.


**Discussion of Model Architecture and Hyperparameters**

Architecture Choices:

● Convolutional Layers: Using 2 convolutional layers per block captures both low- and high-
level features. The 3x3 kernel size is optimal for capturing fine-grained patterns while
maintaining computational efficiency.

● Alternative: Larger kernels (e.g., 5x5) may capture more complex patterns but increase
computation. Smaller kernels (e.g., 1x1) are more computationally efficient but may miss
spatial features.

● Max Pooling: Reduces spatial dimensions and keeps prominent features.

● Alternative: Strided convolutions can also reduce dimensions but may be more
computationally expensive.

● Dropout Regularization: Prevents overfitting by randomly deactivating neurons.

● Alternative: Batch normalization could also be used to stabilize learning and speed up
convergence but adds extra computation.


Hyperparameter Choices:

● Optimizer (Adam): Suited for dynamic learning rates and quick convergence.

● Alternative: SGD with momentum could work but may converge more slowly and require
more tuning.

● Learning Rate Scheduler (ReduceLROnPlateau): Dynamically adjusts the learning rate,
reducing it when the model plateaus.Alternative: Cyclical learning rates could help escape local
minima, but it requires a different setup.

● Early Stopping: Ensures efficient training by halting when no improvement is seen.

● Alternative: Patience could be adjusted to let model train longer or shorter

● Dropout Rates: Gradually increasing dropout in deeper layers helps control overfitting.

● Alternative: Using a constant dropout rate might work but may not be as effective in
deeper layers.


It is clear that the training and validation loss decreases greatly in the first few epochs and stabilizes
towards the later epochs. Towards the end, we see validation loss increasing slightly. Due to our early
stopping callback, we preserve the model with the lowest validation loss and do not update it when the
validation loss increases.

**Model Performance**

Comparing the performance of the linear model and custom CNN, we see that the custom CNN
performs better in train, validation, and test, accuracy/loss. For the custom CNN we see the model is
100% accurate on our training data but still highly accurate on our validation data with an accuracy of
97.61%. Since we are updating the model based on whether the validation decreased in the training loop,
our testing accuracy is naturally lower. To visualize how the model does on a collection of images, we
pick a random sample.

We see the model does fairly well in its predictions for the ASL hand sign images. However, some hand
signs are more tricky to predict due to the hand's orientation. For example, in the top left corner, the
model predicted “u” instead of “v.” This could be because the hand is turned slightly, making the
distance between the two fingers ambiguous, which is the main difference between a “v” and a “u.”
Based on this, we can see across all the classes which have worse and better predictions.

In this confusion matrix, the numbers represent the occurrences of an actual label and a predicted label.
Overall, the model’s true predictions are fairly high but there are a few classes with some errors. For
example, there is one instance where “o” is confused as “0”, “r” is confused as “1”, and others. In
particular, classes “u” and “z” seem to have the lowest true positive rate with only 5/7 correct
predictions for each. As discussed, some of the issues with the ASL hand signs are that some of the hand
signs are similar to each other, thus we will see some false predictions for hand signs that are very close
in similarity such as “0” and “o” and “u” and “z.”

One challenge we had in our model was reproducibility. Each time we trained the model, there were
instances where the model performed worse or better since neural networks randomly initialize weights
and also perform stochastic gradient descent. Thus, many iterations had to be run before we could save
the overall best mode to a .pth file. Since we would in reality have to deliver a model that can be trained
and produce the same results, additional work is needed to achieve that.

To compare our model, we can use one of the most well-known models for object classification:
ResNet-18. This is a pre-trained CNN model that is 18 layers deep and has been trained on millions of
images from the Imagenet database. Using this model, we can see how our custom CNN stacks up.

We see the ResNet model performs better for validation accuracy by ~2% and test accuracy by ~3%.
Though our CNN doesn’t perform as well as ResNet, it is very close in accuracy. With some additional
fine-tuning, we could achieve a model close to this state-of-the-art model. One feature that ResNet uses
that we did not cover in class is residual blocks. These are skip-connection blocks that learn residual
functions regarding the layer inputs, instead of learning unreferenced functions. To improve our model
to make its performance closer to ResNet, we could utilize this feature.

Since CNNs perform very well on ASL hand sign images, we could utilize this in business in various
ways.

**Deployment**

**Use Scenario**

Customer Service: An ASL recognition system can be integrated at customer touchpoints, such
as retail kiosks and website/mobile chat to foster communication with customers, which will lead
to higher satisfaction and loyalty. Customers can access a business’ customer service through
video chat; by signing their needs, the ASL recognition system will be able to capture each sign
and translate them into words. This will enable direct immediate communication with customer
support representatives.

Healthcare (Telemedicine): Healthcare providers and institutions can utilize an ASL recognition
system to better serve patients and their families who are hearing-troubled.
Smart Devices: An ASL recognition system can enhance the functionality of smart devices by
enabling gesture-based interfaces. Smart home devices like lights, thermostats, and security
systems can be set up and controlled with ASL gestures.

Entertainment and Media: The entertainment industry can utilize the ASL recognition system by
creating real-time ASL translation for live events, television broadcasts, and online streaming
platforms, ensuring inclusive experiences for audiences with hearing difficulties.
Self-learning: develop software to learn ASL. Like DuoLingo, learners can use their smartphones
to learn ASL by filming themselves. Our system can instantly judge the correctness of a learner's
gestures and provide constructive feedback, creating an interactive and engaging learning
experience that fosters inclusivity and enhances communication accessibility across diverse
communities.

**Ethical Considerations and Risk**

Data Privacy: As business applications of our ASL recognition system require video footage and
hand movement data, we must inform users about what data are being collected, the usage of
their data, and who will have access to it. The data should only be collected and stored with
explicit consent.

Inclusivity: Our current dataset primarily represents hand signs performed by individuals with
lighter skin tones. To put our system into broader business use, we would need to ensure that it is
trained on diverse datasets that reflect a wide range of signing styles, regional dialects, and
cultural variations within ASL to prevent biases and inaccuracies.

**Conclusion**

Our ASL recognition system provides a solution to unlock accessibility and inclusivity across industries.
The custom CNN model demonstrates strong performance in recognizing ASL hand signs, achieving
high accuracy across training, validation, and testing datasets, which can then be further deployed across
customer service, telemedicine, entertainment, learning platforms, and more.
