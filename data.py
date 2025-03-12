import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2

class DataLoader():
    """
    Data management for labeled image tensorflow datasets.
    """
    
    def __init__(self,batch_size,image_size,shuffle=True,scale=True):
        self.scale = scale
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
        if self.scale:
            dataset = dataset.map(DataLoader.scale_im)
            self.im_type = 'float32'
            print("Images scaled to [0,1].")
        self.dataset = dataset
        
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

        print("CURRENT DATASET SCALE:", self.scale),
        print("IMAGE TYPE:", self.im_type)
        
        plt.figure(figsize=(20,6))
        counter = 0
        for i in range(count):
            for j in range(self.batch_size):
                plt.subplot(count, self.batch_size, i * self.batch_size + j + 1)
                plt.imshow(imgs[counter])
                plt.title(f"{lbls[counter]} {imgs[counter].shape} {imgs[counter].dtype}")
                plt.axis('off')
                counter += 1
        plt.show()


    @staticmethod
    def scale_im(im,label):
        """
        scale image to [0,1]
        """
        im = tf.cast(im, tf.float32)
        im = im / 255
        label = tf.cast(label,tf.float32)
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




class SessionMetrics():
    def __init__(self,metric_save_periot):
        self.d_losses_real = []
        self.d_losses_fake = []
        self.g_losses = []
        self.epochs = []
        self.save_periot = metric_save_periot
    
    def show_outputs(self,d_loss_real,d_loss_fake,g_loss,epoch,interval_test_sample):
        self.d_losses_fake.append(d_loss_fake)
        self.d_losses_real.append(d_loss_real)
        self.g_losses.append(g_loss)
        self.epochs.append(epoch)
        

        # test sample image
        plt.figure(figsize=(12,8))    
        plt.imshow(interval_test_sample); plt.axis("off");plt.title('Test')
        plt.show()

        # graphic of losses
        plt.figure(figsize=(10, 7))
        
        plt.plot(self.epochs, self.d_losses_real, label='D Real Losses', color='blue', marker='o')
        plt.plot(self.epochs, self.d_losses_fake, label='D Fake Losses', color='green', marker='o')
        plt.plot(self.epochs, self.g_losses, label='G Losses', color='red', marker='s')
        plt.title('Loss History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
         
        plt.tight_layout()
        plt.show()
        
        # graphic of losses [0,1]
        plt.figure(figsize=(10, 7))
        
        plt.plot(self.epochs, self.d_losses_real, label='D Real Losses', color='blue', marker='o')
        plt.plot(self.epochs, self.d_losses_fake, label='D Fake Losses', color='green', marker='o')
        plt.plot(self.epochs, self.g_losses, label='G Losses', color='red', marker='s')
        plt.title('Loss History [0-1]')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.ylim(0,1)
        plt.legend()
        plt.grid(True)
         
        plt.tight_layout()
        plt.show()
        
        
    def save_checkpoint(self,
                        epoch,train_topic,
                        d_loss_real,d_loss_fake,g_loss,
                        interval_test_sample,
                        checkpoint_path,
                        outputs_path
                        ):
        if epoch % self.save_periot == 0:
            # metrics save
            f = open(checkpoint_path,'a')
            f.writelines(f"\ntrain:{train_topic} epoch:{epoch} d_loss_real:{d_loss_real:.7f} d_loss_fake:{d_loss_fake:.7f} g_loss:{g_loss:.7f}")
            f.close()

            # sample save
            cv2.imwrite(f'{outputs_path}/{train_topic}_{epoch}.jpg',cv2.cvtColor(interval_test_sample,cv2.COLOR_BGR2RGB))
        else:
            print("Passed metric save.")
            
    def load_old_metrics(self,checkpoints_path):
        f = open(checkpoints_path,"r")
        lines = f.readlines()
        f.close()
        
        for line in lines:
            metrics = line[:-2].split(" ")
            for i,metric in enumerate(metrics):
                a,b = metric.split(":")
                
                if i == 0:
                    pass
                elif i ==1:
                    self.epochs.append(int(b))
                elif i ==2:
                    self.d_losses_real.append(float(b))
                elif i ==3:
                    self.d_losses_fake.append(float(b))
                elif i==4:
                    self.g_losses.append(float(b))
                
            
        
