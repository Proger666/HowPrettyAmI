from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from keras import backend as K
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D
import tensorflow as tf
K.clear_session()

class OnlyOne(object):
    class __OnlyOne:
        def __init__(self):
            self.val = None


            self.model =create_model()
            self.model._make_predict_function()
            self.session = K.get_session()
            self.session.run(tf.global_variables_initializer())

            self.default_graph = tf.get_default_graph()


        def __str__(self):
            return repr(self) + self.val

        def make_prediction(self, img):
            with self.session.as_default():
                with self.default_graph.as_default():
                    self.model.load_weights('phase2')
                    score = self.model.predict(img)
                    return score

    instance = None
    def __new__(cls): # __new__ always a classmethod
        if not OnlyOne.instance:
            OnlyOne.instance = OnlyOne.__OnlyOne()
        return OnlyOne.instance
    def __getattr__(self, name):
        return getattr(self.instance, name)
    def __setattr__(self, name):
        return setattr(self.instance, name)

def create_model():
        model = ResNet50(include_top=False, pooling='avg')
        new_model = Sequential()
        new_model.add(model)
        new_model.add(Dense(1, ))
        return new_model