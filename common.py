import logging


# Enable logging
from telegram import ReplyKeyboardMarkup

logging.basicConfig(format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)
                    # filename='Ketolog')

logger = logging.getLogger(__name__)


class ConsentKeyboard:
    CONSENT_KEY_YES ='Yeah, Science! 👍'
    CONSENT_KEY_NO = 'No, I\'m Luddit 👎'

yes_no_keyboard_markup = ReplyKeyboardMarkup([[ConsentKeyboard.CONSENT_KEY_YES, ConsentKeyboard.CONSENT_KEY_NO]],
                                             one_time_keyboard=True, resize_keyboard=True)



