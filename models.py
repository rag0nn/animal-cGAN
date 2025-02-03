
from tensorflow.keras.layers import Dense, Flatten,Input, Conv2D, UpSampling2D,Conv2DTranspose ,MaxPooling2D, Dropout,Embedding,Reshape,Concatenate # type: ignore
from tensorflow.keras.models import Model,load_model # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.initializers import he_normal# type: ignore
import os

class Models():
    
    def build_models(self,img_shape,num_classes,latent_dim):
        self.loss = 'binary_crossentropy'
        self.opt_disc = Adam(0.0002, 0.5)
        self.opt_gen = Adam(0.0004, 0.5)
        
        self.img_shape = img_shape
        self.discriminator = Models.build_discriminator(self.img_shape,num_classes)
        self.discriminator.compile(loss=self.loss, optimizer=self.opt_disc)
        self.generator = Models.build_generator(latent_dim,num_classes)
        self.gan = Models.build_gan(self.generator,self.discriminator,latent_dim)
        self.gan.compile(loss=self.loss, optimizer=self.opt_gen)
        return self.discriminator,self.generator,self.gan
    
    def build_via_transfer(self,gen_save_path,disc_save_path,latent_dim,g_model_name,d_model_name):
        """
        gen_path = gen_save_path + '/' + os.listdir(gen_save_path)[-1]
        disc_path = disc_save_path + '/' + os.listdir(disc_save_path)[-1]
        """
        gen_path = gen_save_path + '/' + g_model_name
        disc_path = disc_save_path + '/' + d_model_name
        print("Loading generator: ",gen_path)
        print('Loading Discriminator: ',disc_path)    
        self.discriminator = load_model(disc_path)
        self.generator = load_model(gen_path)
        print("Loaded Discriminator:",disc_path)
        print("Loaded Generator",gen_path)
        
        self.gan = Models.build_gan(self.generator,self.discriminator,latent_dim)
        print("GAN Built")
    
        self.loss = 'binary_crossentropy'
        self.opt_disc = Adam(0.0002, 0.5)
        self.opt_gen = Adam(0.0004, 0.5)
        
        self.discriminator.compile(loss=self.loss, optimizer=self.opt_disc)
        self.gan.compile(loss=self.loss, optimizer=self.opt_gen)
        
        return self.discriminator,self.generator,self.gan
    
    def build_discriminator(img_shape,num_classes):
        # image input
        img_input = Input(shape=img_shape)
        # label input
        label_input = Input(shape=(1,))
        label_embedding = Embedding(num_classes, 20)(label_input)
        label_embedding = Dense(64*64*1)(label_embedding)
        label_embedding = Reshape((64,64,1))(label_embedding)
        
        
        x = Conv2D(4, kernel_size=3, strides=2,padding="same" ,kernel_initializer=he_normal())(img_input)
        x = Conv2D(8, kernel_size=3, strides=2,padding="same" ,kernel_initializer=he_normal())(x)
        x = Conv2D(16, kernel_size=3, strides=2,padding="same" ,kernel_initializer=he_normal())(x)
        merged_input = Concatenate()([x, label_embedding])
        x = Conv2D(32, kernel_size=3, strides=2,padding="same" ,kernel_initializer=he_normal())(x)
        x = Conv2D(64, kernel_size=3, strides=2,padding="same" ,kernel_initializer=he_normal())(x)
        x = Conv2D(64, kernel_size=3, strides=2,padding="same" ,kernel_initializer=he_normal())(merged_input)
        x = Flatten()(x)
        x = Dropout(0.25)(x)
        x = Dense(1, activation='sigmoid')(x)
        model = Model([img_input, label_input], x, name="discriminator")
        return model
        
        
    #disc = build_discriminator((512,512,3),3)
    #print(disc.summary())

    def build_generator(latent_dim,num_classes):
        noise_input = Input(shape=(latent_dim,))
        label_input = Input(shape=(1,))

        label_embedding = Embedding(num_classes, 20)(label_input)
        label_embedding = Dense(latent_dim)(label_embedding)
        label_embedding = Reshape((latent_dim,))(label_embedding)
        
        merged_input = Concatenate()([noise_input, label_embedding])
        
        x = Dense(4*4*32)(merged_input)
        x = Reshape((4,4,32))(x)
        x = Conv2DTranspose(64, kernel_size=3, strides=2,padding="same", kernel_initializer=he_normal())(x)
        x = Conv2DTranspose(64, kernel_size=3, strides=2,padding="same", kernel_initializer=he_normal())(x)
        x = Conv2DTranspose(32, kernel_size=3, strides=2,padding="same", kernel_initializer=he_normal())(x)
        x = Conv2DTranspose(16, kernel_size=3, strides=2,padding="same", kernel_initializer=he_normal())(x)
        x = Conv2DTranspose(8, kernel_size=3, strides=2,padding="same", kernel_initializer=he_normal())(x)
        x = Conv2DTranspose(4, kernel_size=3, strides=2,padding="same", kernel_initializer=he_normal())(x)
        x = Conv2DTranspose(3, kernel_size=3, strides=2,padding="same", kernel_initializer=he_normal())(x)
        return Model([noise_input, label_input], x, name="generator")

    #gen = build_generator(100,3)
    #print(gen.summary())


    def build_gan(generator, discriminator, latent_dim):
        noise_input = Input(shape=(latent_dim,))
        label_input = Input(shape=(1,))

        generated_image = generator([noise_input, label_input])

        discriminator.trainable = False

        validity = discriminator([generated_image, label_input])

        
        model = Model([noise_input, label_input], validity, name="gan")
        

        return model

    def show_discriminator(self):
        return self.discriminator.summary()
    
    def show_generator(self):
        return self.generator.summary()
    
    def show_gan(self):
        return self.gan.summary()
    
    def test_generator(self,noise,label):
        """
        Test the generator with noise and label
        :input:
            noise: noise input : 100 int
            label: label input : 0,1,2 int
        """
        return self.generator.predict([noise, label])
    
