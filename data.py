import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2

class DataLoader():
    """
    Data management for labeled image tensorflow datasets.
    """
    
    def __init__(self,batch_size=4,shuffle=True,image_size=(512,512),scaled=True):
        self.scaled = scaled
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.image_size = image_size
        self.im_type = 'uint8'

    
    def load_data(self,dataset_path):
        """
        Load dataset from given path.
        :input
            :param dataset_path: Path to the dataset folder.
            :param batch_size: Batch size for the dataset.
            :param shuffle: Shuffle the dataset.
            :param image_size: Image size for the dataset.
        :return: Loaded dataset.
        """
        print("Loading data from: ", dataset_path)
        dataset = tf.keras.utils.image_dataset_from_directory(
            dataset_path,
            labels="inferred",  # Klasör isimlerinden otomatik label ataması yap
            label_mode="int",   # Etiketleri tam sayı (0, 1, 2, ...) olarak tut
            batch_size=self.batch_size,      # Batch boyutu
            image_size=self.image_size,  
            shuffle=self.shuffle        # Rastgele karıştır
        )
        print("Data loaded successfully.")
        if self.scaled:
            dataset = dataset.map(DataLoader.scale_im)
            self.im_type = 'float32'
            print("Images scaled to [0,1].")
        self.dataset = dataset
        return dataset
    
    def show_sample(self,count=1):
        """
        Get sample from the dataset.
        :input
            :param count: Number of batch samples to show.
        :return: Shown Sample images and labels
        """
        imgs,lbls = [],[]
        for images, labels in self.dataset.take(count):
            images = images.numpy()
            labels = labels.numpy()
            imgs.append(images)
            lbls.append(labels)
        imgs = np.array(imgs)
        lbls = np.array(lbls)
        imgs = np.reshape(imgs, (imgs.shape[0]*imgs.shape[1], imgs.shape[2], imgs.shape[3], imgs.shape[4]))
        lbls = np.reshape(lbls, (lbls.shape[0]*lbls.shape[1]))

        print("CURRENT DATASET SCALE:", self.scaled),
        print("IMAGE TYPE:", self.im_type)
        
        plt.figure(figsize=(20, 4))
        counter = 0
        for i in range(count):
            for j in range(self.batch_size):
                plt.subplot(count, self.batch_size, i * self.batch_size + j + 1)
                plt.imshow(imgs[counter])
                plt.title(f"{lbls[counter]} {imgs[counter].shape} {imgs[counter].dtype}")
                plt.axis('off')
                counter += 1
        plt.show()
        return imgs,lbls

    @staticmethod
    def scale_im(im,label):
        """
        scale image to [0,1]
        """
        im = tf.cast(im, tf.float32)
        im = im / 255
        return im,label

    @staticmethod
    def descale_im(im,label):
        """
        descale image to [0,255]
        """
        im = im * 255
        im = im.astype(np.uint8)

        min_val = im.min()
        max_val = im.max()

        if min_val < 0 or max_val > 255:    
            # 0 ile 255 arasında ölçekleme yap
            im = ((im - min_val) / (max_val - min_val) * 255).astype('uint8')
            
        return im,label




class Process():
    def __init__(self):
        self.d_losses_real = []
        self.d_losses_fake = []
        self.g_losses = []
    
    def show_outputs(self,d_loss_real,d_loss_fake,g_loss,interval_test_sample):
        self.d_losses_fake.append(d_loss_fake)
        self.d_losses_real.append(d_loss_real)
        self.g_losses.append(g_loss)
        
        epochs_history = range(len(self.d_losses_real))

        # test sample image
        plt.figure(figsize=(10,7))    
        plt.imshow(interval_test_sample); plt.axis("off");plt.title('Test')
        plt.show()

        # graphic of losses
        plt.figure(figsize=(10, 7))
        
        plt.plot(epochs_history, self.d_losses_real, label='D Real Losses', color='blue', marker='o')
        plt.plot(epochs_history, self.d_losses_fake, label='D Fake Losses', color='green', marker='o')
        plt.plot(epochs_history, self.g_losses, label='G Losses', color='red', marker='s')
        plt.title('Loss History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
         
        plt.tight_layout()
        plt.show()
        
        
    def save_checkpoint(self,
                        epoch,train_topic,
                        g_model,d_model,
                        d_loss_real,d_loss_fake,g_loss,
                        interval_test_sample,
                        g_save_path,d_save_path,
                        checkpoint_path,
                        outputs_path
                        ):
        # models save
        g_model.save(g_save_path+f"/g_model_{train_topic}_{epoch}.h5")
        d_model.save(d_save_path+f"/d_model_{train_topic}_{epoch}.h5")

        # metrics save
        f = open(checkpoint_path,'a')
        f.writelines(f"\ntrain:{train_topic} epoch:{epoch} d_loss_real:{d_loss_real:.7f} d_loss_fake:{d_loss_fake:.7f} g_loss:{g_loss:.7f}")
        f.close()

        # sample save
        cv2.imwrite(f'{outputs_path}/{train_topic}_{epoch}.jpg',interval_test_sample)