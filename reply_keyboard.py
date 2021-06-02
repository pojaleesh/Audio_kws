import telebot


class Keyboard:
    def __init__(self, bot):
        self.bot = bot

    def choose_emotion_model(self, message):
        emotion_markup = telebot.types.ReplyKeyboardMarkup(True, False)
        emotion_markup.row('Back')
        emotion_markup.row('CNN_1')
        emotion_markup.row('CNN_2')
        self.bot.send_message(
            message.from_user.id, 
            'Choose a model for emotion and gender recognition', 
            reply_markup=emotion_markup
        )

    def choose_spotter_model(self, message):
        spotter_markup = telebot.types.ReplyKeyboardMarkup(True, False)
        spotter_markup.row('Back')
        spotter_markup.row('DNN')
        spotter_markup.row('DS_CNN')
        spotter_markup.row('CNN')
        spotter_markup.row('ATT_RNN')
        spotter_markup.row('CRNN')
        self.bot.send_message(
            message.from_user.id, 
            'Choose a model for spotter recognition', 
            reply_markup=spotter_markup
        )

    def main_menu(self, message):
        main_markup = telebot.types.ReplyKeyboardMarkup(True, False)
        main_markup.row("Choose a spotter model")
        main_markup.row("Choose a emotion and gender model")
        main_markup.row("Get current models")
        self.bot.send_message(message.from_user.id, 'Pick models:', reply_markup=main_markup)
