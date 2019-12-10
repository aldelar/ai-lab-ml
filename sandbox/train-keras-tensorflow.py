#
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image

from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

# azureml-core of version 1.0.72 or higher is required
# azureml-contrib-dataset of version 1.0.72 or higher is required
from azureml.core import Workspace, Datastore, Run
from azureml.contrib.dataset import Dataset
import azureml.dataprep.native

# load train and test set into numpy arrays
# note we scale the pixel intensity values to 0-1 (by dividing it with 255.0) so the model can converge faster.

subscription_id = os.environ['AZURE_SUBSCRIPTION_ID']
resource_group = 'ai-lab'
workspace_name = 'ailabml'
workspace = Workspace(subscription_id, resource_group, workspace_name)

# get dataset
ds = Dataset.get_by_name(workspace, name='Light Bulbs-2019-12-08 00:35:33')
df = ds.to_pandas_dataframe()

# prepare training data

# Images
train_image = []
for i in tqdm(range(df.shape[0])):
    si = df.loc[i].image_url.to_pod()
    if i == 0:
        datastore = Datastore.get(workspace, si['arguments']['datastoreName'])
    datastore.download(target_path='.',prefix=si['resourceIdentifier'],overwrite=True,show_progress=False)
    img = image.load_img(si['resourceIdentifier'], target_size=(28,28,1), grayscale=False)
    img = image.img_to_array(img)
    img = img/255
    train_image.append(img)
X = np.array(train_image)

# Labels
f = np.vectorize(lambda i: sum(bytearray(i[0].encode(encoding='UTF-8')))) # convert label => Int
y = to_categorical(f(df['label'].values))

# generate train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=12, test_size=0.2)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape, sep = '\n')

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(28,28,3)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1074, activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

print('Predict the test set')
y_hat = model.predict(X_test)

# calculate accuracy on the prediction
acc = np.average(y_hat == y_test)
print('Accuracy is', acc)

# get hold of the current run
run = Run.get_context()
run.log('accuracy', np.float(acc))

os.makedirs('outputs', exist_ok=True)
# note file saved in the outputs folder is automatically uploaded into experiment record
joblib.dump(value=model, filename='outputs/ai_lab_defect_detection_model.pkl')