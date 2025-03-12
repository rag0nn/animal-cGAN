
from tensorflow.keras.layers import Dense, Flatten,Input, Conv2D, UpSampling2D,BatchNormalization,Conv2DTranspose ,MaxPooling2D, Dropout,Embedding,Reshape,Concatenate # type: ignore
from tensorflow.keras.models import Model,load_model # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.initializers import he_normal# type: ignore
import numpy as np
import pickle
from tensorflow.keras.losses import BinaryCrossentropy # type: ignore

class Models():
    
    def __init__(self,img_shape,num_classes,latent_dim):
        self.img_shape = img_shape
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.discriminator = None
        self.generator = None
        self.gan = None
        
    def build_models(self):
        self.discriminator = self.build_discriminator()
        self.generator = self.build_generator()
        self.gan = self.build_gan(self.generator,self.discriminator)
        self.compile_models()
        
        self.gan.summary()
    
    def build_discriminator(self):
        # image input
        img_input = Input(shape=self.img_shape)
        # label input
        label_input = Input(shape=(1,))
        label_embedding = Embedding(self.num_classes, 20)(label_input)
        label_embedding = Dense(8*8*1)(label_embedding)
        label_embedding = Reshape((8,8,1))(label_embedding)
        
        x = Conv2D(48, kernel_size=4, strides=2,padding="same" )(img_input)
        x = BatchNormalization()(x)
        x = Conv2D(48, kernel_size=4, strides=2,padding="same" )(x)
        x = BatchNormalization()(x)
        x = Conv2D(48, kernel_size=4, strides=2,padding="same" )(x)
        x = BatchNormalization()(x)
        x = Conv2D(4, kernel_size=4, strides=2,padding="same" )(x)    
        
        x = Concatenate()([x, label_embedding])
        x = Flatten()(x)
        
        x = Dropout(0.25)(x)
        x = Dense(10, activation='relu')(x)
        x = Dropout(0.25)(x)
        x = Dense(1, activation='sigmoid')(x)
        model = Model([img_input, label_input], x, name="discriminator")
        return model
    
    def build_generator(self):
        noise_input = Input(shape=(self.latent_dim,))
        noise_embedding = Dense(8*8*3)(noise_input)
        noise_embedding = Reshape((8,8,3))(noise_embedding)
        
        label_input = Input(shape=(1,))
        label_embedding = Embedding(self.num_classes, 20)(label_input)
        label_embedding = Dense(8*8*3)(label_embedding)
        label_embedding = Reshape((8,8,3))(label_embedding)
        
        merged_input = Concatenate()([noise_embedding, label_embedding])
        
        x = Conv2DTranspose(64, kernel_size=4, strides=2,padding="same", )(merged_input)
        x = BatchNormalization()(x)
        x = Conv2DTranspose(64, kernel_size=4, strides=2,padding="same", )(x)
        x = BatchNormalization()(x)
        x = Conv2DTranspose(64, kernel_size=4, strides=2,padding="same", )(x)
        x = BatchNormalization()(x)
        x = Conv2DTranspose(3, kernel_size=4, strides=2,padding="same", )(x)
        return Model([noise_input, label_input], x, name="generator")

    def build_gan(self,generator, discriminator):
        noise_input = Input(shape=(self.latent_dim,))
        label_input = Input(shape=(1,))

        generated_image = generator([noise_input, label_input])

        discriminator.trainable = False
        validity = discriminator([generated_image, label_input])
        model = Model([noise_input, label_input], validity, name="gan")
        
        return model

    def test_generator(self,noise,label):
        """
        Test the generator with noise and label
        :input:
            noise: noise input : 100 int
            label: label input : 0,1,2 int
        """
        assert self.generator is not None, "Generator not defined"
        return self.generator.predict([noise, label])
    
    def save_models(self,train_topic,epoch,gen_save_path,disc_save_path,gan_opt_save_path,disc_opt_save_path):
        gen_path = f"{gen_save_path}/{train_topic}-{epoch}.h5"
        disc_path = f"{disc_save_path}/{train_topic}-{epoch}.h5"
        gan_opt_path = f"{gan_opt_save_path}/{train_topic}-{epoch}.npy"
        disc_opt_path = f"{disc_opt_save_path}/{train_topic}-{epoch}.npy"
        
        self.generator.save_weights(gen_path)
        self.discriminator.save_weights(disc_path)

        with open(gan_opt_path, "wb") as f:
            pickle.dump(self.gan.optimizer.get_weights(), f)
        with open(disc_opt_path, "wb") as f:
            pickle.dump(self.discriminator.optimizer.get_weights(), f)
        """
        np.save(disc_opt_path, self.discriminator.optimizer.get_weights())
        np.save(gan_opt_path, self.gan.optimizer.get_weights())
        """
        print("Models saved successfully","\n",gen_path,"\n",disc_path,"\n",gan_opt_path,"\n",disc_opt_path)

    
    def load_models(self,gen_model_path,disc_model_path,gan_opt_path,disc_opt_path):
        self.discriminator = self.build_discriminator()
        self.generator = self.build_generator()
        self.generator.load_weights(gen_model_path)
        self.discriminator.load_weights(disc_model_path)
        self.gan = self.build_gan(self.generator,self.discriminator)
        self.discriminator.optimizer.set_weights(np.load(gan_opt_path, allow_pickle=True))
        self.gan.optimizer.set_weights(np.load(disc_opt_path, allow_pickle=True))
        print("Models loaded successfully")
        
    def show_learning_params(self):
        assert self.generator is not None,"Generator not defined"
        assert self.discriminator is not None,"Discriminator not defined"
        assert self.gan is not None,"GAN not defined"       
        print("Discriminator learning rate: ",self.discriminator.optimizer.learning_rate)
        print("GAN learning rate: ",self.gan.optimizer.learning_rate)
    
    def compile_models(self,gan_lr=0.0002,disc_lr=0.000001,loss='binary_crossentropy'):
        assert self.generator is not None,"Generator not defined"
        assert self.discriminator is not None,"Discriminator not defined"
        assert self.gan is not None,"GAN not defined"
        loss_ = BinaryCrossentropy(label_smoothing=0.1)
        self.discriminator.compile(loss=loss_, optimizer=Adam(disc_lr, 0.5))
        self.gan.compile(loss=loss, optimizer=Adam(gan_lr, 0.5)) 
        print("Compile Process Ended.")
        self.show_learning_params()
        

