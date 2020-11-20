# HSE project. Implementation of speech processing machine learning algorithms: emotion recognition, biometric, wake up word recognition.

### Emotion and biometric recognition

The goal is to determine emotion and biometrics by voice message.

#### Datasets

- SAVEE dataset. https://www.kaggle.com/ejlok1/surrey-audiovisual-expressed-emotion-savee
- RAVDESS dataset. https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio
- TESS dataset. https://www.kaggle.com/ejlok1/toronto-emotional-speech-set-tess
- CREMA-D dataset. https://www.kaggle.com/ejlok1/cremad

![data_hist](https://user-images.githubusercontent.com/55574235/99854703-8aa0d780-2baf-11eb-9be7-95f9ef9983a0.png)

#### Baseline Conv1D models

Take the mean in the columns of MFCC

##### Model 1

- Conv1D → ReLU → BatchNormalization → Flatten → Dense → Softmax
- Optimizer - RMSprop, lr = 0.0001

![model_1](https://user-images.githubusercontent.com/55574235/99855311-c5efd600-2bb0-11eb-931a-a28f0237d2de.png)

##### Model 2

- Conv1D → ReLU → BatchNormalization → Flatten → Dense → Softmax
- Optimizer - Adam, lr = 0.0001

![model_2](https://user-images.githubusercontent.com/55574235/99855420-08b1ae00-2bb1-11eb-9dda-06a59a201453.png)

##### Model 3

- Conv1D → ReLU → Conv1D → BatchNormalization → ReLU → Dropout(0.25) → MaxPool1D → Conv1D → ReLU → Conv1D → ReLU → Conv1D → ReLU → Conv1D → BatchNormalization → ReLU → Dropout(0.25) → MaxPool1D → Conv1D → ReLU → Conv1D → ReLU → Flatten → Dense → Softmax
- Optimizer - RMSprop, lr = 0.00001

![model_3](https://user-images.githubusercontent.com/55574235/99855835-e40a0600-2bb1-11eb-9a46-418f914a2dae.png)
