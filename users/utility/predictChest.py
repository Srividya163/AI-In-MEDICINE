import numpy as np
import pandas as pd
from keras.utils import load_img
from keras.models import load_model

labels = ['Cardiomegaly', 
          'Emphysema', 
          'Effusion', 
          'Hernia', 
          'Infiltration', 
          'Mass', 
          'Nodule', 
          'Atelectasis',
          'Pneumothorax',
          'Pleural_Thickening', 
          'Pneumonia', 
          'Fibrosis', 
          'Edema', 
          'Consolidation']

def start_process(imagepath):
    import os
    from django.conf import settings
    img_path = os.path.join(settings.MEDIA_ROOT,imagepath)
    model_path = os.path.join(settings.MEDIA_ROOT,'ChestModel.h5')
    model = load_model(model_path,compile=False)
    print('*'*50)
    print(model)
    
    df_path = os.path.join(settings.MEDIA_ROOT,'nih'+'/'+'train-small.csv')
    df = pd.read_csv(df_path)
    
    def get_mean_std_per_batchR(image_path, df, H=320, W=320):
        sample_data = []
        for idx, img in enumerate(df.sample(100)["id"].values):
            # path = image_dir + img
            sample_data.append(
                np.array(load_img(image_path, target_size=(H, W))))

        mean = np.mean(sample_data[0])
        std = np.std(sample_data[0])
        return mean, std
    
    def load_imageR(img_path, df, preprocess=True, H=320, W=320):
        """Load and preprocess image."""
        mean, std = get_mean_std_per_batchR(img_path, df, H=H, W=W)
        x = load_img(img_path, target_size=(H, W))
        if preprocess:
            x -= mean
            x /= std
            x = np.expand_dims(x, axis=0)
        return x
    
    
    
    preprocessed_input = load_imageR(img_path,  df)
    predictions = model.predict(preprocessed_input)
    prediction = np.argmax(predictions)
    prediction = labels[prediction]
    print('-'*50)
    print(prediction)
    return prediction
    