from tensorflow.keras import Model
from tensorflow.keras.layers import Dropout, GlobalAveragePooling2D, TimeDistributed, Conv2D, BatchNormalization, MaxPool2D, ConvLSTM2D, MaxPool3D, LayerNormalization, Dense, Flatten, GRU 


class Encoder(Model):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv3d_1 = TimeDistributed(Conv2D(2, kernel_size = (7,7), padding = 'same'))
        self.batchnorm_1 = TimeDistributed(BatchNormalization())
        self.batchnorm_2 = TimeDistributed(BatchNormalization())
        self.batchnorm_3 = TimeDistributed(BatchNormalization())
        self.maxpool3d_1 = TimeDistributed(MaxPool2D(pool_size=(2,2)))
        self.maxpool3d_2 = TimeDistributed(MaxPool2D(pool_size=(2,2)))
        self.maxpool3d_3 = TimeDistributed(MaxPool2D(pool_size=(2,2)))
        self.conv3d_2 = TimeDistributed(Conv2D(4, kernel_size = (5,5), padding = 'same'))
        self.conv3d_3 = TimeDistributed(Conv2D(8, kernel_size = (3,3), padding = 'same'))
        # self.td = TimeDistributed()


    def call(self, inputs):
        print(inputs.shape)
        
        x = self.conv3d_1(inputs)
        x = self.batchnorm_1(x)
        x = self.maxpool3d_1(x)
        x = self.conv3d_2(x)
        x = self.batchnorm_2(x)
        x = self.maxpool3d_2(x)
        x = self.conv3d_3(x)
        x = self.batchnorm_3(x)
        x = self.maxpool3d_3(x)
        return x

class GRU_Model(Model):
    
    def __init__(self):
        super(GRU_Model, self).__init__()
        # self.encoder = TimeDistributed(Encoder())
        self.encoder = Encoder()
        self.flt = TimeDistributed(GlobalAveragePooling2D())
        self.gru_1 = GRU(32, return_sequences=False)
        # self.layernorm = LayerNormalization()
        # self.maxpool2d = MaxPool2D()
        # self.flatten = Flatten()
        self.dense_1 = Dense(10, activation = 'relu')
        self.drp = Dropout(0.5)
        self.classifier = Dense(6, activation = 'softmax') 

        
    def call(self, inputs):
        # inputs = inputs.reshape(inputs[0], inputs[1], inputs[2], inputs[3], 1)
        print(inputs.shape)
        x = self.encoder(inputs)
        x = self.flt(x)
        x = self.gru_1(x)
        # x = self.layernorm(x)
        # x = self.maxpool2d(x)
        # x = self.flatten(x)
        x = self.dense_1(x)
        x = self.drp(x)

        return self.classifier(x)