# Introduction
Face recognition is a method of identifying or verifying the identity of an individual using their face. Face recognition systems can be used to identify people in photos, video, or in real-time. It involves comparing one input image to all images in an image library in order to determine who the input image belongs to. Or if it does not belong to the database at all.

### How Facial Recognition Works
#### Step 1: Face detection
The camera detects and locates the image of a face, either alone or in a crowd. The image may show the person looking straight ahead or in profile.

#### Step 2: Face analysis
Next, an image of the face is captured and analyzed by a neural network algorithm. The program examines the physical features of an individual’s face to distinguish uniqueness from others. Key factors include the distance between your eyes, the depth of your eye sockets, the distance from forehead to chin, the shape of your cheekbones, and the contour of the lips, ears, and chin. The aim is to identify the facial landmarks that are key to distinguishing your face.

#### Step 3: Converting the image to data
The face capture process transforms analog information (a face) into a set of digital information (data) based on the person's facial features. Your face's analysis is essentially turned into a mathematical formula. The numerical code is called a faceprint.

#### Step 4: Finding a match
Your faceprint is then compared against a database of other known faces. For example, the FBI has access to up to 650 million photos, drawn from various state databases

One key factor that can strongly affect the effectiveness of facial recognition is lighting. In order for facial recognition to work, it’s very important to have good lighting to clearly show all of the individual’s facial features.
The system would have a lot of difficulties identifying the individual. It’s also possible that the picture in the system that is used to match with the individual is outdated, meaning that the system would likely be less accurate in identifying. Regardless of these obstacles, facial recognition is still one of the most accurate methods of identifying an individual and holds a lot of potential for the future.

# Libraries Used 

## Keras 
Keras is an open-source software library that provides a Python interface for artificial neural networks. Keras acts as an interface for the TensorFlow library. Keras contains numerous implementations of commonly used neural-network building blocks such as layers, objectives, activation functions, optimizers, and a host of tools to make working with image and text data easier to simplify the coding necessary for writing deep neural network code. In addition to standard neural networks, Keras has support for convolutional and recurrent neural networks.

## Open CV
OpenCV (Open Source Computer Vision Library) is a library of programming functions mainly aimed at real-time computer vision.

# Working Explained 

### 1. Collecting the Images (face detection)
* **1.1 Creating Environment for Python 3.6**
* **1.2 HAAR CASCADE Classfiers**

<img src="https://www.bogotobogo.com/python/OpenCV_Python/images/FaceDetection/HaarFeatures.png">

OpenCV already contains many pre-trained classifiers for face, eyes, smile etc. Those XML files are stored in opencv/data/haarcascades/ folder. In my Project I have used the **haarcascade_frontalface_default.xml** to detect the frontal face view of a person. 

* **1.3 Def Face_Extractor**

defining a function named Face Extractor. It detects the faces and returns a cropped image. 

**faces = face_classifier.detectMultiScale(img, 1.3, 5)**

1. Scale Factor : 1.3 , it means we're using a small step for resizing, i.e. reduce size by 30 %, we increase the chance of a matching size with the model for detection is found, while it's expensive. 
2. minNeighbors : Parameter specifying how many neighbors each candidate rectangle should have to retain it. This parameter will affect the quality of the detected faces: higher value results in less detections but with higher quality. We're using 5 in the code.

* **1.4 Start the Webcam and Take Images**

Keep on clicking the number of images as well as simultaneoulsy applying the Face_Extractor function to crop and adjust them. Further save these images in a specified directory with a path name (being decided by a loop) .
We also have cv2.waitkey set to 1 , inorder to keep on clicking frame images until the loop final value is reached. Then Destroy all windows. 

### 2. Face Recognision 
* **resize the Images**
* **Divide the images clicked in the last step into 2 folders named - Train and Test**
* **VGG16 -**
is a convolution neural net (CNN ) architecture which was used to win ILSVR(Imagenet) competition in 2014. It is considered to be one of the excellent vision model architecture till date. Most unique thing about VGG16 is that instead of having a large number of hyper-parameter they focused on having convolution layers of 3x3 filter with a stride 1 and always used same padding and maxpool layer of 2x2 filter of stride 2. It follows this arrangement of convolution and max pool layers consistently throughout the whole architecture. In the end it has 2 FC(fully connected layers) followed by a softmax for output. The 16 in VGG16 refers to it has 16 layers that have weights. This network is a pretty large network and it has about 138 million (approx) parameters. A CNN has Convolutional layers which can detect Patterns (a particular part of an image). Even deeper layers can then detect full objects. Here we have used VGG16 which has 16 layers. 
Here we have first counted the number of classes of our images (categories) using the "glob" function. Then we have flattened the images and then added them to the VGG16 as a final layer.
* **Create the final Model -**
* **Generate More Images -**
Here we use the 'ImageDataGenerator' from Keras.Preprocessor.image . For Training data we change Shear Range , Zoom range and Horizontal Flip to generate images. For the Testing Data , we only Rescale the images (by div them by 255)
* **Create a Training and Testing Set**
* **Fit a Model using the Epoch Approach -**
Where Epoach is a hyperparameter that defines the number times that the learning algorithm will work through the entire training dataset.
* **Check the Face Recognision on a Video Feed through the Webcam -**
OpenCV can be used to check the live feed with the help of our webcam. Then I have further taken frames from this feed and then applied those frames/images in our model to check if the face being detected in the images is any one of the matching ones our model has been trained with or not. The Prediction is done on the basis of the location of the folder i.e. if 0th folder images match the image being captured in the frame then the output will be the name of the 0th folder.

# Results 
Succesfully able to Recognise Faces from the Live feed through the Webcam. 
Final Accuracy - 73% 

# Scope For Improvement 
* VGG19 can be used for more layers and better detection 
* The Model will predict as many people as you supply (number of folders and images) but the computational time will be high. Therefore to improve it , we might need a better computation system or a GPU.

# Applications of this Model 
* Security companies are using facial recognition to secure their premises.
* Immigration checkpoints use facial recognition to enforce smarter border control.
* Fleet management companies can use face recognition to secure their vehicles.
* Ride-sharing companies can use facial recognition to ensure the right passengers are picked up by the right drivers.
* IoT benefits from facial recognition by allowing enhanced security measures and automatic access control at home.
* Law Enforcement can use facial recognition technologies as one part of AI-driven surveillance systems.
* Retailers can use facial recognition to customize offline offerings and to theoretically map online purchasing habits with their online ones.
* Google incorporates the technology into Google Photos and uses it to sort pictures and automatically tag them based on the people recognized.

Face recognition is an emerging technology that can provide many benefits. Face recognition can save resources and time, and even generate new income streams, for companies that implement it right. It has come a long way in the last twenty years. Today, machines are able to automatically verify identity information for secure transactions, for surveillance and security tasks, and for access control to buildings etc. It’s difficult to be certain. Some experts predict that our faces will replace IDs, passports and credit card pin numbers. Given the fact how convenient and cost-effective this technology is, this prediction is not far-fetched. If this prediction becomes a reality, any company that implemented the technology today might gain a competitive advantage in the future.¶

# Bibliography 
* https://www.youtube.com/watch?v=S6NR8GdXxTE&t=727s
* https://keras.io/api/applications/vgg/
* https://en.wikipedia.org/wiki/Keras
* https://en.wikipedia.org/wiki/OpenCV
* https://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_Image_Object_Detection_Face_Detection_Haar_Cascade_Classifiers.php
* https://towardsdatascience.com/step-by-step-vgg16-implementation-in-keras-for-beginners-a833c686ae6c
