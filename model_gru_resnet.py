from tensorflow.keras import Model
from tensorflow.keras.layers import Dropout, GlobalAveragePooling2D, TimeDistributed, Conv2D, BatchNormalization, MaxPool2D, ConvLSTM2D, MaxPool3D, LayerNormalization, Dense, Flatten, GRU 
from tensorflow.keras.applications import ResNet152V2


class Encoder(Model):
    def __init__(self):
        super(Encoder, self).__init__()
        self.resnet = TimeDistributed(ResNet152V2(include_top=False, weights='imagenet'))


    def call(self, inputs):
        x= self.resnet(inputs)
        return x

class GRU_Resnet_Model(Model):
    
    def __init__(self):
        super(GRU_Resnet_Model, self).__init__()
        self.encoder = Encoder()
        self.flt = TimeDistributed(GlobalAveragePooling2D())
        self.gru_1 = GRU(32, return_sequences=False)
        self.dense_1 = Dense(10, activation = 'relu')
        self.drp = Dropout(0.5)
        self.classifier = Dense(6, activation = 'softmax') 

        
    def call(self, inputs):
        self.encoder.trainable= False
        x = self.encoder(inputs)
        x = self.flt(x)
        x = self.gru_1(x)
        # x = self.layernorm(x)
        # x = self.maxpool2d(x)
        # x = self.flatten(x)
        x = self.dense_1(x)
        x = self.drp(x)

        return self.classifier(x)