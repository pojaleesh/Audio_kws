# HSE project. Implementation of speech processing machine learning algorithms: emotion recognition, biometric, wake up word recognition.

## Emotion and biometric recognition

The goal is to determine emotion and biometrics by voice message.

### Datasets

- SAVEE dataset. https://www.kaggle.com/ejlok1/surrey-audiovisual-expressed-emotion-savee
- RAVDESS dataset. https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio
- TESS dataset. https://www.kaggle.com/ejlok1/toronto-emotional-speech-set-tess
- CREMA-D dataset. https://www.kaggle.com/ejlok1/cremad

![data_hist](https://user-images.githubusercontent.com/55574235/99854703-8aa0d780-2baf-11eb-9be7-95f9ef9983a0.png)

### MFCC

#### Preprocessing
Let's split the entire track into small time intervals - frames. Moreover, frames can overlap each other. <br/>
Let's determine frame like a vector: <br/>
![formula](https://render.githubusercontent.com/render/math?math=%24x%5Bk%5D%2C%20k%20%5Cin%20%5B1%3B%20N%5D%24) <br/>
The first of all, we'll expand frame via FFT: <br/>
![formula](https://render.githubusercontent.com/render/math?math=%24X%5Bk%5D%20%3D%20%5Csum_%7Bn%3D1%7D%5EN%20x%5Bn%5D%5Ccdot%20e%5E%7B2%5Cpi%20ikn%20%2F%20N%7D%2C%200%20%5Cleqslant%20k%20%3C%20N%24) <br/>
After that, let's make Hamming window to flatten the values: <br/>
![formula](https://render.githubusercontent.com/render/math?math=%24H%5Bk%5D%20%3D%200.54%20-%200.46%5Ccdot%5Ccos(2%5Cpi%20k%2F%20(N-1))%2C%200%20%5Cleqslant%20k%20%3C%20N%24) <br/>
![formula](https://render.githubusercontent.com/render/math?math=%24X%5Bk%5D%20%3D%20X%5Bk%5D%20%5Ccdot%20H%5Bk%5D%2C%200%20%5Cleqslant%20k%20%3C%20N%24) <br/>
![Voice_waveform_and_spectrum](https://user-images.githubusercontent.com/55574235/100019481-52480600-2e08-11eb-8087-163704a05f49.png)

#### Mel-Filters

Direct - transform(Mel from frequency): <br/>
![formula](https://render.githubusercontent.com/render/math?math=%24g(f)%20%3D%201127%5Ccdot%5Clog(1%20%2B%20f%2F700)%24) <br/>
Inverse - transform(Frequency from mel): <br/>
![formula](https://render.githubusercontent.com/render/math?math=%24g%5E%7B-1%7D(m)%20%3D%20700%5Ccdot(e%5E%7Bm%2F1127%7D%20-%201)%24) <br/>
![formula](https://render.githubusercontent.com/render/math?math=%24H_m%20%3D%0A%5Cbegin%7Bcases%7D%0A0%2C%20k%20%3C%20m%20%5C%5C%0A%5Cdfrac%7Bk%20-%20f%5Bm-1%5D%7D%7Bf%5Bm%5D%20-%20f%5Bm-1%5D%7D%2C%20f%5Bm-1%5D%5Cleqslant%20k%20%3C%20f%5Bm%5D%20%5C%5C%0A%5Cdfrac%7Bf%5Bm%2B1%5D%20-%20k%7D%7Bf%5Bm%2B1%5D%20-%20f%5Bm%5D%7D%2C%20f%5Bm%5D%20%5Cleqslant%20k%20%5Cleqslant%20f%5Bm%2B1%5D%20%5C%5C%0A0%2C%20k%20%3E%20f%5Bm%2B1%5D%0A%5Cend%7Bcases%7D%24) <br/>
![formula](https://render.githubusercontent.com/render/math?math=%24f%5Bm%5D%20%3D%20%5CBig(%5Cdfrac%7BN%7D%7BF_S%7D%5CBig)%5Ccdot%20g%5E%7B-1%7D%5CBig(g(f_1)%20%2B%20m%5Ccdot%5Cdfrac%7Bg(f_h)%20-%20g(f_1)%7D%7BM%20%2B%201%7D%5CBig)%24)

#### Get the energy for each frame

![formula](https://render.githubusercontent.com/render/math?math=%24S%5Bm%5D%20%3D%20ln(%5Csum_%7Bk%3D0%7D%5E%7Bn-1%7D%20X%5Bk%5D%5E2%5Ccdot%20H_m%5Bk%5D)%2C%200%20%5Cleqslant%20m%20%3C%20N%24)

#### Get MFCC

![formula](https://render.githubusercontent.com/render/math?math=%24c%5Bn%5D%20%3D%20%5Csum_%7Bm%3D0%7D%5E%7BM-1%7D%20S%5Bm%5D%5Ccdot%5Ccos(%5Cpi%20n(m%20%2B%201%2F2)%2FM)%2C%200%20%5Cleqslant%20n%20%3C%20M%24)


### Baseline Conv1D models

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

### Conv2D models

##### Model 1

- Conv2D → BatchNormalization → ReLU → MaxPool2D → Dropout(0.2) - x4 - → Flatten → Dense → Dropout(0.2) → BatchNormalization → ReLU → Dropout(0.2)
- Optimizer - Adam, lr = 0.001
- Optimizer - Adam, lr = 0.01

##### Model 2

- Conv2D → BatchNormalization → ReLU → MaxPool2D - x4 - → Flatten → Dense → Dropout(0.2) → BatchNormalization → ReLU → Dropout(0.2)
- Optimizer - Adam, lr = 0.001

###### Comparison Model 1 and Model 2

![comparison_1](https://user-images.githubusercontent.com/55574235/103152833-e0bbf880-47b5-11eb-8782-5fed8c5c486a.png)
