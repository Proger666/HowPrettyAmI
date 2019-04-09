#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random
import sys
from time import sleep

import pandas as pd
from keras import Sequential
from keras.layers import Dense
from telegram import ReplyKeyboardMarkup, ReplyKeyboardRemove, ChatAction
import face_recognition
import numpy as np
from keras.preprocessing import image
from scipy import stats
import uuid
from common import logger, yes_no_keyboard_markup, ConsentKeyboard
from model.DB import init_db
from model.tetnsor import OnlyOne
from state_machine import *
import pydal
import sys
import matplotlib.pyplot as plt
import skimage
from PIL import Image
import PIL
import tensorflow as tf
import cv2
from mtcnn.mtcnn import MTCNN

global graph, new_model
new_model = OnlyOne()

sys.path.insert(0, '')
from telegram.ext import (Updater, CommandHandler, MessageHandler, Filters, RegexHandler,
                          ConversationHandler)

import configparser
import json
from tensorflow.python.keras.applications import ResNet50


####### Internal ########
def error(bot, update, error):
    """Log Errors caused by Updates"""
    logger.warning('Update "%s" caused error "%s"', update, error)


def startCommand(bot, context):
    '''Present to user initial info and keyboard to choose'''

    # Dump user ID to DB
    bot.send_chat_action(chat_id=context.message.chat_id,
                         action=ChatAction.TYPING)
    context.message.reply_text(
        "Hello stranger!\nI'm clever bot which can say how beautiful you are!\n\nJust send me nud...picture of your face and see result.")
    sleep(2)
    bot.send_chat_action(chat_id=context.message.chat_id,
                         action=ChatAction.TYPING)
    context.message.reply_text(
        "I will use your picture for science - to better understand how humans differs from each other.\n\nDo you agree?",
        reply_markup=yes_no_keyboard_markup,
        parse_mode='Markdown'
    )
    return CONSENT


def parse_consent(bot, context):
    '''Parse yes or no answer for consent'''

    if context.message.text == ConsentKeyboard.CONSENT_KEY_NO:
        bot.send_chat_action(chat_id=context.message.chat_id,
                             action=ChatAction.TYPING)
        context.message.reply_text('Ok, I will delete your image upon finish.\n *now send me your pic*.',
                                   parse_mode='Markdown', reply_markup=ReplyKeyboardRemove())
        return WAIT_FOR_PIC
    elif context.message.text == ConsentKeyboard.CONSENT_KEY_YES:
        bot.send_chat_action(chat_id=context.message.chat_id,
                             action=ChatAction.TYPING)
        context.message.reply_text('Great! I will be better with every step now.\n *now send me your pic*.',
                                   parse_mode='Markdown', reply_markup=ReplyKeyboardRemove())
        return WAIT_FOR_PIC
    else:
        bot.send_chat_action(chat_id=context.message.chat_id,
                             action=ChatAction.TYPING)
        context.message.reply_text("Unfortunately, my overlord forbids me to answer on ambiguous questions :(\n",
                                   reply_markup=yes_no_keyboard_markup,
                                   parse_mode='Markdown')
        return CONSENT


def rescale_img(img):
    im = Image.open(img)
    # rescale img
    im.thumbnail((350, 350), PIL.Image.ANTIALIAS)
    im.save(img)


def find_face(img):
    __img = img
    _image = face_recognition.load_image_file(img)
    faces = face_recognition.face_locations(_image, number_of_times_to_upsample=0, model="cnn")

    if len(faces):
        faces = cascad_face_detect(img)
    sub_face = []
    # Draw a rectangle around the faces
    for (top, right, bottom, left) in faces:
        top = int(top / 2)
        right = int(right * 2)
        bottom = int(bottom * 1.5)
        left = int(left / 2)
        face_image = _image[top:bottom, left:right]
        _face = image.img_to_array(face_image)
        new_fname = __img.replace('user_photo', 'user_photo_' + str(random.randint(1, 10)))
        cv2.imwrite('../pics/' + new_fname, _face)
        rescale_img(new_fname)
        sub_face.append(new_fname)
    rescale_img(__img)

    return sub_face


def cascad_face_detect(_img):
    _img = cv2.imread(_img)
    # convert the test image to gray image as opencv face detector expects gray images
    gray_img = cv2.cvtColor(_img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    # let's detect multiscale (some images may be closer to camera than others) images
    faces = face_cascade.detectMultiScale(
        gray_img,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30, 30)
    )
    if len(faces) == 0:
        faces = face_cascade.detectMultiScale(
            gray_img,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
    return faces


def judje_photo(context, photo_file):
    from scipy import stats

    DIR = "G:\\"
    file = open(DIR + 'ratings.csv', 'r')
    df = pd.read_csv(file)
    from collections import defaultdict
    all_images = defaultdict(list)
    for filename, rating in df[['Filename', 'Rating']].values:
        all_images[filename].append(rating)
    data = {}
    for filename, ratings in all_images.items():
        data[filename] = np.mean(ratings)
    ratings = dict(data)
    from sklearn.preprocessing import MinMaxScaler

    labels = np.array(list(ratings.values()))
    scaler = MinMaxScaler().fit((labels).reshape(-1, 1))
    f_name = '../pics/user_photo{}.jpg'.format(uuid.uuid4())
    img = photo_file.download(f_name)
    faces = find_face(img)
    actual_scores = []
    percentiles = []
    test_Y = scaler.transform(labels.reshape(-1, 1))

    #### model ####
    with tf.Graph().as_default():
        for face in faces:
            face = image.load_img(face)
            face = image.img_to_array(face)
            score = new_model.make_prediction(face.reshape((1,) + face.shape))
            actual_scores.append(scaler.inverse_transform(score.reshape(-1, 1)))
            percentiles.append(stats.percentileofscore(test_Y, score[0][0]))
    return [round(float(x), 1) for x in actual_scores], [round(x, 1) for x in percentiles], faces


def process_image(bot, context):
    '''Find face, rearrange, flatten'''
    # Read the image
    photo_file = context.message.photo[-1].get_file()
    context.message.reply_text('Let me think a bit... Hmmmm..')
    return judje_photo(context, photo_file)


def judje_peasant(bot, context):
    '''Make predictions for photos'''

    scores, percents, faces = process_image(bot, context)
    if len(scores) == 0:
        context.message.reply_text("Can't find face :(\nTry another pic")
        return WAIT_FOR_PIC

    for score, percent, face in zip(scores, percents, faces):
        bot.send_photo(chat_id=context.message.chat_id,
                       photo=open('../pics/' + face, 'rb'))
        context.message.reply_text("I think you are {}/5".format(score))
        context.message.reply_text("Its better than *{}%* ! Congratz".format(percent), parse_mode='Markdown')
        bot.send_chat_action(chat_id=context.message.chat_id,
                             action=ChatAction.TYPING)
        sleep(1)

    context.message.reply_text("Send me another pic.")
    return WAIT_FOR_PIC


def stop(bot, context):
    '''Stop converstation with user'''
    user = context.message.from_user
    logger.error("User %s stoped the conversation.", user.first_name)
    bot.send_chat_action(chat_id=context.message.chat_id,
                         action=ChatAction.TYPING)
    bot.send_message(text='Sayonara!', chat_id=context.message.chat_id,
                     reply_markup=ReplyKeyboardRemove())

    return ConversationHandler.END


def cancel(bot, context):
    '''Stop converstation with user'''
    user = context.message.from_user
    logger.error("User %s canceled current operation.", user.first_name)
    bot.send_chat_action(chat_id=context.message.chat_id,
                         action=ChatAction.TYPING)
    bot.send_message(text='Canceled', chat_id=context.message.chat_id,
                     reply_markup=ReplyKeyboardRemove())

    return startCommand(bot, context)


def main():
    updater = Updater('747503234:AAEZPy4JR5E7cmJIc3ZvZN3j6OFOfrMtroo')
    # Get the dispatcher to register handlers
    dp = updater.dispatcher
    global db
    db = init_db()

    # State machine - Start -> choice> serve request -> end
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', startCommand), MessageHandler(Filters.text, startCommand)],

        states={

            CONSENT: [MessageHandler(Filters.text,
                                     parse_consent),
                      ],

            WAIT_FOR_PIC: [MessageHandler(Filters.photo, judje_peasant)],

        },

        fallbacks=[CommandHandler('stop', stop),
                   CommandHandler('cancel', cancel),
                   MessageHandler(Filters.text,
                                  startCommand)]
    )

    dp.add_handler(conv_handler)

    # log all errors
    dp.add_error_handler(error)

    # Start the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()


if __name__ == '__main__':
    main()
