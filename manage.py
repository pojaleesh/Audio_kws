import threading
import telebot
import requests
import urllib
import subprocess
from wav_work import solve, delete
from predict_emotion import make_predict_emotion
from telebot import types
from Spotter_recognition.predict_spotter import make_predict_spotter
from Spotter_recognition.service.plot_making import make_plot
import tensorflow as tf
import os


token = "1614991779:AAHzVpGl1ng7HmNHm95WymAUq8AAEkO-xaE" 

bot = telebot.TeleBot(token)
model_at_rnn = tf.keras.models.load_model(os.path.join(os.getcwd(), 'Spotter_recognition/saved_models/ATT_RNN.h5'))
model_dnn = tf.keras.models.load_model(os.path.join(os.getcwd(), 'Spotter_recognition/saved_models/DNN.h5')) 
model_cnn = tf.keras.models.load_model(os.path.join(os.getcwd(), 'Spotter_recognition/saved_models/CNN.h5')) 
model_crnn = tf.keras.models.load_model(os.path.join(os.getcwd(), 'Spotter_recognition/saved_models/CRNN.h5')) 
model_ds_cnn = tf.keras.models.load_model(os.path.join(os.getcwd(), 'Spotter_recognition/saved_models/DNN.h5')) 

flags_emotion = {
    'model': None,
    'path': 'temp.wav',
    'mean': 'emotion_data_service/mean',
    'std': 'emotion_data_service/std',
    'labels': 'emotion_data_service/labels',
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

@bot.message_handler(commands=['start'])
def start_message(message):
    bot.send_message(message.chat.id, "Okey, let's start")
    keyboard = types.InlineKeyboardMarkup();
    key_CNN = types.InlineKeyboardButton(text='CNN', callback_data='CNN');
    key_CRNN = types.InlineKeyboardButton(text='CRNN', callback_data='CRNN');
    key_ATT_RNN = types.InlineKeyboardButton(text='ATT_RNN', callback_data='ATT_RNN');
    key_DNN = types.InlineKeyboardButton(text='DNN', callback_data='DNN');
    key_DS_CNN = types.InlineKeyboardButton(text='DS_CNN', callback_data='DS_CNN');
    keyboard.add(key_CNN)
    keyboard.add(key_CRNN)
    keyboard.add(key_ATT_RNN)
    keyboard.add(key_DNN)
    keyboard.add(key_DS_CNN)
    bot.send_message(
        message.from_user.id, 
        text="Pick a model for spotter recognition", 
        reply_markup=keyboard
    )


@bot.message_handler(commands=['spotter_model'])
def start_message(message):
    keyboard = types.InlineKeyboardMarkup();
    key_CNN = types.InlineKeyboardButton(text='CNN', callback_data='CNN');
    key_CRNN = types.InlineKeyboardButton(text='CRNN', callback_data='CRNN');
    key_ATT_RNN = types.InlineKeyboardButton(text='ATT_RNN', callback_data='ATT_RNN');
    key_DNN = types.InlineKeyboardButton(text='DNN', callback_data='DNN');
    key_DS_CNN = types.InlineKeyboardButton(text='DS_CNN', callback_data='DS_CNN');
    keyboard.add(key_CNN)
    keyboard.add(key_CRNN)
    keyboard.add(key_ATT_RNN)
    keyboard.add(key_DNN)
    keyboard.add(key_DS_CNN)
    bot.send_message(
        message.from_user.id,
        text="Pick a model for spotter recognition",
        reply_markup=keyboard
    )

    
@bot.callback_query_handler(func=lambda call: True)
def callback_worker(call):
    global flags_spotter
    if call.data == "CNN":
        flags_spotter['model'] = 'CNN'
        bot.send_message(
            call.message.chat.id, 
            'You pick a CNN model'
        );
        bot.send_message(
            call.message.chat.id, 
            'Here you can read a decription : https://arxiv.org/pdf/1711.07128.pdf'
        )
        bot.send_photo(call.message.chat.id, photo=open('figures/CNN_acc_with_aug.jpg', 'rb'))
    elif call.data == "CRNN":
        flags_spotter['model'] = 'CRNN'
        bot.send_message(
            call.message.chat.id, 
            'You pick a CRNN model'
        );
        bot.send_message(
            call.message.chat.id, 
            'Here you can read a decription : https://arxiv.org/pdf/1711.07128.pdf'
        )
        bot.send_photo(call.message.chat.id, photo=open('figures/CRNN_acc_with_aug.jpg', 'rb'))
    elif call.data == "ATT_RNN":
        flags_spotter['model'] = 'ATT_RNN'
        bot.send_message(
            call.message.chat.id, 
            'You pick a ATT_RNN model'
        );
        bot.send_message(
            call.message.chat.id, 
            'Here you can read a decription : https://arxiv.org/pdf/1808.08929.pdf'
        )
        bot.send_photo(call.message.chat.id, photo=open('figures/ATT_RNN_acc_with_aug.jpg', 'rb'))
    elif call.data == "DNN":
        flags_spotter['model'] = 'DNN'
        bot.send_message(
            call.message.chat.id, 
            'You pick a DNN model'
        );
        bot.send_message(
            call.message.chat.id, 
            'Here you can read a decription : https://arxiv.org/pdf/1711.07128.pdf'
        )
        bot.send_photo(call.message.chat.id, photo=open('figures/DNN_acc_with_aug.jpg', 'rb'))
    elif call.data == "DS_CNN":
        flags_spotter['model'] = 'DS_CNN'
        bot.send_message(
            call.message.chat.id, 
            'You pick a DS_CNN model'
        );
        bot.send_message(
            call.message.chat.id, 
            'Here you can read a decription : https://arxiv.org/pdf/1711.07128.pdf'
        )
        bot.send_photo(call.message.chat.id, photo=open('figures/DS_CNN_acc_with_aug.jpg', 'rb'))


@bot.message_handler(commands=['end'])
def start_message(message):
    bot.send_message(message.chat.id, 'end')


@bot.message_handler(commands=['model'])
def start_message(message):
    global flags_spotter
    spotter_model = flags_spotter['model']
    if spotter_model:
        bot.send_message(message.chat.id, f'Your model for spotter recognition - {spotter_model}')
    else:
        bot.send_message(message.chat.id, f'You did not pick a model for spotter recognition.')
        bot.send_message(message.chat.id, message.from_user.id)


@bot.message_handler(content_types=['text'])
def get(message):
    print('get message')
    bot.send_message(message.chat.id, message.text)


@bot.message_handler(content_types=['voice'])
def get_audio(audio):
    file_id = audio.voice.file_id
    url = f'https://api.telegram.org/bot{token}/getFile?file_id={file_id}'
    res = requests.get(url).json()
    temp = bot.get_file(file_id)
    file_path = temp.file_path
    urll = f'https://api.telegram.org/file/bot{token}/{file_path}'
    urllib.request.urlretrieve(urll, filename='1.oga', reporthook=None, data=None)
    average_amplitude = solve()
    bot.send_message(audio.chat.id, f'average_amplitude={average_amplitude}')
    #bot.send_photo(audio.chat.id, photo=open('plot1.png', 'rb'))
    #bot.send_photo(audio.chat.id, photo=open('plot2.png', 'rb'))
    #src_filename = 'plot1.png'
    #process = subprocess.call(['del', src_filename], shell=True)
    #src_filename = 'plot2.png'
    #process = subprocess.call(['del', src_filename], shell=True)
    _, res = make_predict_emotion(flags_emotion)
    gender, emotion = res.split('_')
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
            make_plot(prediction_data, 0.95)
            bot.send_photo(audio.chat.id, photo=open('model_prediction.jpg', 'rb'));
    delete()


bot.polling()
