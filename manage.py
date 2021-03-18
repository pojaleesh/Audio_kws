import telebot
import requests
import urllib
import subprocess
from wav_work import solve, delete
from predict_emotion import make_predict


token = '1292656407:AAGKUjDoTHKEYpWAc8hi_mu-i7zUizbFgME'
bot = telebot.TeleBot('1292656407:AAGKUjDoTHKEYpWAc8hi_mu-i7zUizbFgME')


@bot.message_handler(commands=['start'])
def start_message(message):
    bot.send_message(message.chat.id, 'start')


@bot.message_handler(commands=['end'])
def start_message(message):
    bot.send_message(message.chat.id, 'end')


@bot.message_handler(content_types=['text'])
def get(message):
    print('get message')
    bot.send_message(message.chat.id, message.text)


@bot.message_handler(content_types=['voice'])
def get_audio(audio):
    print('get voice')
    file_id = audio.voice.file_id
    url = f'https://api.telegram.org/bot{token}/getFile?file_id={file_id}'
    res = requests.get(url).json()
    temp = bot.get_file(file_id)
    file_path = temp.file_path
    urll = f'https://api.telegram.org/file/bot{token}/{file_path}'
    urllib.request.urlretrieve(urll, filename='1.oga', reporthook=None, data=None)
    average_amplitude = solve()
    print(average_amplitude)
    bot.send_message(audio.chat.id, f'average_amplitude={average_amplitude}')
    bot.send_photo(audio.chat.id, photo=open('plot1.png', 'rb'))
    bot.send_photo(audio.chat.id, photo=open('plot2.png', 'rb'))
    src_filename = 'plot1.png'
    process = subprocess.call(['del', src_filename], shell=True)
    src_filename = 'plot2.png'
    process = subprocess.call(['del', src_filename], shell=True)
    res = make_predict()
    bot.send_message(audio.chat.id, f'gender_emotion = {res}')
    delete()


@bot.message_handler(content_types=['document'])
def get_audio(audio):
    print('get document')


bot.polling()
