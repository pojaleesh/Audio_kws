import telebot
from collections import defaultdict


class User2Model():
    
    def __init__(self, bot):
        self.user2model_emotion = defaultdict(str)
        self.user2model_spotter = defaultdict(str)
        self.bot = bot

    def get_emotion_model(self, user_id):
        if user_id in self.user2model_emotion:
            return self.user2model_emotion[user_id]
        else:
            return None

    def get_spotter_model(self, user_id):
        if user_id in self.user2model_spotter:
            return self.user2model_spotter[user_id]
        else:
            return None

    def set_emotion_model(self, user_id, emotion_model):
        self.user2model_emotion[user_id] = emotion_model

    def set_spotter_model(self, user_id, spotter_model):
        self.user2model_spotter[user_id] = spotter_model





