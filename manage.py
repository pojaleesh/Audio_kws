import threading
import telebot
import requests
import urllib
import subprocess
import multiprocessing
from wav_work import solve, delete
from Emotion_biometric_recognition.predict_emotion import make_predict_emotion
from telebot import types
from Spotter_recognition.predict_spotter import make_predict_spotter
from Spotter_recognition.service.plot_making import make_plot
import tensorflow as tf
import os
import sys
import time
from rq import Queue
from redis import Redis
from reply_keyboard import Keyboard

token = "1825121446:AAGxTdrkzyvhOisEXDH2p2nmVkVe9STVtyE" 

bot = telebot.TeleBot(token)
keyboard = Keyboard(bot)

model_at_rnn = tf.keras.models.load_model(os.path.join(os.getcwd(), 'Spotter_recognition/saved_models/ATT_RNN.h5'))
model_dnn = tf.keras.models.load_model(os.path.join(os.getcwd(), 'Spotter_recognition/saved_models/DNN.h5')) 
model_cnn = tf.keras.models.load_model(os.path.join(os.getcwd(), 'Spotter_recognition/saved_models/CNN.h5')) 
model_crnn = tf.keras.models.load_model(os.path.join(os.getcwd(), 'Spotter_recognition/saved_models/CRNN.h5')) 
model_ds_cnn = tf.keras.models.load_model(os.path.join(os.getcwd(), 'Spotter_recognition/saved_models/DS_CNN.h5')) 

emotion_cnn_1 = tf.keras.models.load_model(
    os.path.join(os.getcwd(), 'Emotion_biometric_recognition/saved_models/CNN1_emotion.h5')
)

emotion_cnn_2 = tf.keras.models.load_model(
    os.path.join(os.getcwd(), 'Emotion_biometric_recognition/saved_models/CNN2_emotion.h5')
)

flags_emotion = {
    'model': None,
    'path': 'temp.wav',
    'sr': 44100,
    'mfcc': {
        'window_size': 40.0,
        'stride': 20.0,
        'upper_frequency_limit': 7600,
        'lower_frequency_limit': 20,
        'sample_rate': 44100,
        'filterbank_channel_count': 40,
        'dct_coefficient_count': 20,
        'audio_len': 314818
    },
    'load_data': {
        'labels': 'emotion_data_service/labels',
        'mean': 'emotion_data_service/mean',
        'std': 'emotion_data_service/std'
    },
    'shift': 22050,
    'frame_lenght': 314818,
    'path': 'temp.wav',
}

flags_spotter = {
    'model': None,
    'path': 'temp.wav',
    'sr': 16000,
    'mfcc_type': 'tensorflow',
    'frame_lenght': 6400,
    'shift': 1600,
    'mfcc': {
        'window_size': 40.0,
        'stride': 20.0,
        'upper_frequency_limit': 7600,
        'lower_frequency_limit': 20,
        'sample_rate': 16000,
        'filterbank_channel_count': 40,
        'dct_coefficient_count': 20
    },
    'load_data': {
        'labels': 'spotter_data_service/labels',
    },
}

redis_conn = Redis()
queue = Queue(connection=redis_conn)


@bot.message_handler(commands=['start'])
def get_message(message):
    main_markup = telebot.types.ReplyKeyboardMarkup(True, False)
    main_markup.row("Choose a spotter model")
    main_markup.row("Choose a emotion and gender model")
    main_markup.row("Get current models")
    bot.send_message(message.from_user.id, 'Pick models:', reply_markup=main_markup)


@bot.message_handler(func=lambda mess: "Menu" == mess.text, content_types=['text'])
def handle_text(message):
    keyboard.main_menu(message)


@bot.message_handler(func=lambda mess: 'Choose a spotter model' == mess.text, content_types=['text'])
def get_message(message):
    keyboard.choose_spotter_model(message)


@bot.message_handler(func=lambda mess: 'Choose a emotion and gender model' == mess.text, content_types=['text'])
def get_message(message):
    keyboard.choose_emotion_model(message)


@bot.message_handler(func=lambda mess: 'Back' == mess.text, content_types=['text'])
def get_message(message):
    keyboard.main_menu(message)


@bot.message_handler(func=lambda mess: 'CNN' == mess.text, content_types=['text'])
def get_message(message):
    global flags_spotterdd
    flags_spotter['model'] = 'CNN'
    bot.send_message(
        message.chat.id,
        'You pick a CNN model'
    );
    bot.send_message(
        message.chat.id,
        'Here you can read a decription : https://arxiv.org/pdf/1711.07128.pdf'
    )
    bot.send_photo(message.chat.id, photo=open('figures/CNN_acc_with_aug.jpg', 'rb'))


@bot.message_handler(func=lambda mess: 'CRNN' == mess.text, content_types=['text'])
def get_message(message):
    global flags_spotter
    flags_spotter['model'] = 'CRNN'
    bot.send_message(
        message.chat.id,
        'You pick a CRNN model'
    );
    bot.send_message(
        message.chat.id,
        'Here you can read a decription : https://arxiv.org/pdf/1711.07128.pdf'
    )
    bot.send_photo(message.chat.id, photo=open('figures/CRNN_acc_with_aug.jpg', 'rb'))


@bot.message_handler(func=lambda mess: 'ATT_RNN' == mess.text, content_types=['text'])
def get_message(message):
    global flags_spotter
    flags_spotter['model'] = 'ATT_RNN'
    bot.send_message(
        message.chat.id,
        'You pick a ATT_RNN model'
    );
    bot.send_message(
        message.chat.id,
        'Here you can read a decription : https://arxiv.org/pdf/1808.08929.pdf'
    )
    bot.send_photo(message.chat.id, photo=open('figures/ATT_RNN_acc_with_aug.jpg', 'rb'))


@bot.message_handler(func=lambda mess: 'DNN' == mess.text, content_types=['text'])
def get_message(message):
    global flags_spotter
    flags_spotter['model'] = 'DNN'
    bot.send_message(
        message.chat.id,
        'You pick a DNN model'
    );
    bot.send_message(
        message.chat.id,
        'Here you can read a decription : https://arxiv.org/pdf/1711.07128.pdf'
    )
    bot.send_photo(message.chat.id, photo=open('figures/DNN_acc_with_aug.jpg', 'rb'))


@bot.message_handler(func=lambda mess: 'DS_CNN' == mess.text, content_types=['text'])
def get_message(message):
    global flags_spotter
    flags_spotter['model'] = 'DS_CNN'
    bot.send_message(
        message.chat.id,
        'You pick a DS_CNN model'
    );
    bot.send_message(
        message.chat.id,
        'Here you can read a decription : https://arxiv.org/pdf/1711.07128.pdf'
    )
    bot.send_photo(message.chat.id, photo=open('figures/DS_CNN_acc_with_aug.jpg', 'rb'))


@bot.message_handler(func=lambda mess: 'CNN_1' == mess.text, content_types=['text'])
def get_message(message):
    global flags_spotter
    flags_emotion['model'] = 'CNN_1'
    bot.send_message(
        message.chat.id,
        'You pick a CNN_1 model'
    );
    bot.send_message(
        message.chat.id,
        'Here you can read a decription : https://arxiv.org/pdf/1803.03759.pdf'
    )
    bot.send_photo(message.chat.id, photo=open('figures/CNN1_emotion_accuracy.jpg', 'rb'))


@bot.message_handler(func=lambda mess: 'CNN_2' == mess.text, content_types=['text'])
def get_message(message):
    global flags_spotter
    flags_emotion['model'] = 'CNN_2'
    bot.send_message(
        message.chat.id,
        'You pick a CNN_2 model'
    );
    bot.send_message(
        message.chat.id,
        'Here you can read a decription : https://arxiv.org/pdf/1803.03759.pdf'
    )
    bot.send_photo(message.chat.id, photo=open('figures/CNN2_emotion_accuracy.jpg', 'rb'))


@bot.message_handler(func=lambda mess: 'Get current models' == mess.text, content_types=['text'])
def get_message(message):
    global flags_spotter
    spotter_model = flags_spotter['model']
    emotion_model = flags_emotion['model']
    if spotter_model:
        bot.send_message(message.chat.id, f'Your model for spotter recognition - {spotter_model}')
    else:
        bot.send_message(message.chat.id, f'You did not pick a model for spotter recognition.')
        #bot.send_message(message.chat.id, message.from_user.id)
    if emotion_model:
        bot.send_message(message.chat.id, f'Your model for emotion and gender recognition - {emotion_model}')
    else:
        bot.send_message(message.chat.id, f'You did not pick a model for emotion recognition.')
        #bot.send_message(message.chat.id, message.from_user.id)


@bot.message_handler(content_types=['voice'])
def get_audio(audio):
    delete()
    file_id = audio.voice.file_id
    url = f'https://api.telegram.org/bot{token}/getFile?file_id={file_id}'
    res = requests.get(url).json()
    temp = bot.get_file(file_id)
    file_path = temp.file_path
    urll = f'https://api.telegram.org/file/bot{token}/{file_path}'
    urllib.request.urlretrieve(urll, filename='1.oga', reporthook=None, data=None)
    job = queue.enqueue(solve)
    time.sleep(3)
    average_amplitude = job.result
    bot.send_message(audio.chat.id, f'average_amplitude={average_amplitude}')
    bot.send_photo(audio.chat.id, photo=open('plot1.png', 'rb'))
    bot.send_photo(audio.chat.id, photo=open('plot2.png', 'rb'))
    src_filename = 'plot1.png'
    process = subprocess.call(['del', src_filename], shell=True)
    src_filename = 'plot2.png'
    process = subprocess.call(['del', src_filename], shell=True)
    prediction_emotion = None
    if flags_emotion['model'] == 'CNN_1':
        prediction_emotion = make_predict_emotion(emotion_cnn_1, flags_emotion)
    elif flags_emotion['model'] == 'CNN_2':
        prediction_emotion = make_predict_emotion(emotion_cnn_2, flags_emotion)
    else:
        bot.send_message(audio.chat.id, 'You did not pick a model for emotion and gender recognition :(')
    if prediction_emotion:
        gender, emotion = prediction_emotion.split('_')
        bot.send_message(audio.chat.id, f'Your gender - {gender}, your emotion - {emotion}')
    indicator, temp, ans = None, None, None
    if flags_spotter['model'] == 'CRNN':
        indicator, prediction_data, ans = make_predict_spotter(model_crnn, flags_spotter, 0.95)
    elif flags_spotter['model'] == 'CNN':
        indicator, prediction_data, ans = make_predict_spotter(model_cnn, flags_spotter, 0.95)
    elif flags_spotter['model'] == 'ATT_RNN':
        indicator, prediction_data, ans = make_predict_spotter(model_at_rnn, flags_spotter, 0.95)
    elif flags_spotter['model'] == 'DNN':
        indicator, prediction_data, ans = make_predict_spotter(model_dnn, flags_spotter, 0.95)
    elif flags_spotter['model'] == 'DS_CNN':
         indicator, prediction_data, ans = make_predict_spotter(model_ds_cnn, flags_spotter, 0.95)
    else:
        bot.send_message(audio.chat.id, 'You did not pick a model for spotter recognition :(')
    if ans != None:
        if len(ans) == 0:
            bot.send_message(audio.chat.id, "Unfortunatelly model did'n recognize any spotter words")
        else:
            bot.send_message(audio.chat.id, ans)
            job = queue.enqueue(make_plot, prediction_data, 0.95)
            time.sleep(4)
            bot.send_photo(audio.chat.id, photo=open('model_prediction.jpg', 'rb'));
    delete()


bot.polling()
