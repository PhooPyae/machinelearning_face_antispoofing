# Fae Anti-spoofing using Eyes Movement and CNN-based Liveness Detection

Face Anti-spoofing is about preventing spoofing attack on the system like face recognition for example. Attakers will try to use fake images, printed image and video replays to attack the system.

This project is about the anti-spoofing system for a system like face recognition.
### Summarization of the project
The project has three components:
1. Eyes movement detection
2. Patch-based CNN
3. Depth-based CNN (further work)

These components can be assumed as streamline of the system. In the meantime, it is the combination of eyes movement and patch-based cnn.

![system architecture](https://user-images.githubusercontent.com/20230956/120360173-77c14980-c32e-11eb-8c20-42e004384135.png)

### Eyes Movemet Detection

![eyes movement](https://user-images.githubusercontent.com/20230956/120360709-0fbf3300-c32f-11eb-8f40-6a4c93df3180.png)

Challenge-Response approach is mainly used.
The main idea behind it is that the system generates 3 random challenges and the user needs to act the challenges correctly. This helps to verify the live user action.


### Patch-based CNN

![patch-based](https://user-images.githubusercontent.com/20230956/120361527-ef43a880-c32f-11eb-9832-a3be91060463.png)

Instead of using the whole face image as training data, here I used  patches of face, like the distinct features of a face for example eyes, nose, lips, eyebow.
This idea is to make the model to learn more details of the features.
##### SIFT(Scale Invariant Feature Transform) is used to extract the key points with its corresponding description.
##### CNN (Convolutional Neural Network) to learn the feature and to classify whether the unknown data is live or spoof.

You can read more here : https://ieeexplore.ieee.org/document/8921091/






