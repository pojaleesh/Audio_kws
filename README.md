# HSE project. Implementation of speech processing machine learning algorithms: emotion recognition, biometric, wake up word recognition.

## Emotion and biometric recognition

The goal is to determine emotion and biometrics by voice message.

### Datasets

- SAVEE dataset. https://www.kaggle.com/ejlok1/surrey-audiovisual-expressed-emotion-savee
- RAVDESS dataset. https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio
- TESS dataset. https://www.kaggle.com/ejlok1/toronto-emotional-speech-set-tess
- CREMA-D dataset. https://www.kaggle.com/ejlok1/cremad

![data_hist](https://user-images.githubusercontent.com/55574235/99854703-8aa0d780-2baf-11eb-9be7-95f9ef9983a0.png)

### Data representation. MFCC

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

### Conv2D models

##### Architecture

![CNN_emotion_architecture](https://user-images.githubusercontent.com/55574235/120560046-bf1d0800-c40a-11eb-9511-d7e527a5d5e7.png)

##### CNN_1

![CNN_1_acc](https://user-images.githubusercontent.com/55574235/120560410-56825b00-c40b-11eb-8c66-c317a72c2296.png)
![CNN_1_loss](https://user-images.githubusercontent.com/55574235/120560276-1ae79100-c40b-11eb-8084-0f2d9ab0e456.png)

##### CNN_2

![CNN_2_acc](https://user-images.githubusercontent.com/55574235/120560308-2a66da00-c40b-11eb-9445-fb9e3875b04a.png)
![CNN_2_loss](https://user-images.githubusercontent.com/55574235/120560351-38b4f600-c40b-11eb-8e48-415fc4aca0d4.png)

## Wake up words(Key words) recognition

The goal is to highlight key words in audio.

### Datasets

- Tensorflow dataset. https://arxiv.org/abs/1804.03209

The labels words, that i need to predict in are : yes, no, up, down, left, right, on, off, stop, go.

![data_hist_1](https://user-images.githubusercontent.com/55574235/103959642-78d4bd00-517a-11eb-8741-32d13473121f.png)
![data_hist_2](https://user-images.githubusercontent.com/55574235/103959663-84c07f00-517a-11eb-9a71-011165493fae.png)

### DNN model

<img width="514" alt="Снимок экрана 2021-06-03 в 01 41 27" src="https://user-images.githubusercontent.com/55574235/120561231-d0ffaa80-c40c-11eb-92af-7ba651f5ca78.png">

![DNN_acc](https://user-images.githubusercontent.com/55574235/120560778-08218c00-c40c-11eb-8bba-d1cd3bc296b0.png)
![DNN_loss](https://user-images.githubusercontent.com/55574235/120560798-12dc2100-c40c-11eb-9f07-25144d919b83.png)

### CNN model

<img width="647" alt="Снимок экрана 2021-06-03 в 01 42 32" src="https://user-images.githubusercontent.com/55574235/120561314-f7bde100-c40c-11eb-9051-39c0b039c7e5.png">

![CNN_acc](https://user-images.githubusercontent.com/55574235/120561409-28057f80-c40d-11eb-8625-bbfd0a81ff58.png)
![CNN_loss](https://user-images.githubusercontent.com/55574235/120561429-30f65100-c40d-11eb-8c49-c98ea5f0b517.png)

### CRNN model

<img width="347" alt="Снимок экрана 2021-06-03 в 01 44 45" src="https://user-images.githubusercontent.com/55574235/120561487-47041180-c40d-11eb-9981-1707351b477a.png">

![CRNN_acc](https://user-images.githubusercontent.com/55574235/120561497-4cf9f280-c40d-11eb-9d41-130efe2e310e.png)
![CRNN_loss](https://user-images.githubusercontent.com/55574235/120561514-55eac400-c40d-11eb-885f-8e3ed3107193.png)

### DS_CNN model

<img width="324" alt="Снимок экрана 2021-06-03 в 01 46 12" src="https://user-images.githubusercontent.com/55574235/120561592-7b77cd80-c40d-11eb-8c57-830f93dfcadb.png">

![DS_CNN_acc](https://user-images.githubusercontent.com/55574235/120561610-816dae80-c40d-11eb-9273-89cd7187ed91.png)
![DS_CNN_loss](https://user-images.githubusercontent.com/55574235/120561617-8599cc00-c40d-11eb-8bbd-72bc7757138c.png)

### ATT_RNN model

<img width="784" alt="Снимок экрана 2021-06-03 в 01 46 57" src="https://user-images.githubusercontent.com/55574235/120561643-964a4200-c40d-11eb-9578-cc2c834f33d0.png">

![ATT_RNN_acc](https://user-images.githubusercontent.com/55574235/120561661-9d715000-c40d-11eb-864a-78bdfe5e68d8.png)
![ATT_RNN_loss](https://user-images.githubusercontent.com/55574235/120561679-a2ce9a80-c40d-11eb-80fe-7ae0f3bcd4d2.png)

## Augmentations

Use theese augmentations from https://github.com/iver56/audiomentations

|  Augmentation      | Chance of use on sample batch |
| ---------------- | --------------------- |
| TimeStretch | 0.3 |
| PitchShift | 0.3 |
| Shift | 0.3 |
| ClippingDistortion | 0.3 |
| [TimeMask](https://www.isca-speech.org/archive/Interspeech_2019/abstracts/2680.html) | 0.5 |
| [FrequencyMask](https://www.isca-speech.org/archive/Interspeech_2019/abstracts/2680.html) | 0.5 | 
| Background noise | 0.66 | 

## Results

|  Model name      | Result  |
| ---------------- | --------------------- | 
|[dnn](https://arxiv.org/pdf/1711.07128.pdf) | 89.0% |
|[cnn](https://arxiv.org/pdf/1711.07128.pdf) |  96.4% |
|[crnn](https://arxiv.org/pdf/1711.07128.pdf) | 97% |
|[ds_cnn](https://arxiv.org/pdf/1711.07128.pdf) | 96.9% |
|[att_rnn](https://arxiv.org/pdf/1808.08929.pdf) | 97.4%  |


## Making a prediction

We will move uniformly with a constant window and build predictions for each window.
![Predcit](https://user-images.githubusercontent.com/55574235/120563200-aca5cd00-c410-11eb-887a-fb136e163eac.png)
As a result we have list:
<img src="https://latex.codecogs.com/svg.image?[(p_1,&space;label_1),&space;\ldots,&space;(p_n,&space;label_n)]" title="[(p_1, label_1), \ldots, (p_n, label_n)]" /> <br/>
This list we can represented as a plot: <br/>
<img width="355" alt="Снимок экрана 2021-06-03 в 02 12 31" src="https://user-images.githubusercontent.com/55574235/120563401-28a01500-c411-11eb-925b-dc4ef442dd0e.png">

## Testing 

https://t.me/SpotterRecognitionBot








