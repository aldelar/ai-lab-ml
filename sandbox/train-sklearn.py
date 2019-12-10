#
import argparse
import os
import numpy as np
from tqdm import tqdm

from sklearn import svm, metrics, datasets
from sklearn.utils import Bunch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

from skimage.io import imread
from skimage.transform import resize

from azureml.core import Run, Workspace, Datastore
from azureml.contrib.dataset import Dataset

def load_image_files(dimension=(256, 256)):
    
    subscription_id = os.environ['AZURE_SUBSCRIPTION_ID']
    resource_group = 'ai-lab'
    workspace_name = 'ailabml'
    workspace = Workspace(subscription_id, resource_group, workspace_name)

    # get dataset (online run)
    #run = Run.get_context()
    #dataset = run.input_datasets['Light Bulbs-2019-12-08 00:35:33']

    # get dataset (offline run)
    ds = Dataset.get_by_name(workspace, name='Light Bulbs-2019-12-08 00:35:33')
    df = ds.to_pandas_dataframe()

    # Images
    descr = "Defect Detection Dataset"
    images = []
    flat_data = []
    target = []
    categories = set()
    for i in tqdm(range(df.shape[0])):
        si = df.loc[i].image_url.to_pod()
        if i == 0:
            datastore = Datastore.get(workspace, si['arguments']['datastoreName'])
        categories.add(df.loc[i].label[0])
        datastore.download(target_path='.',prefix=si['resourceIdentifier'],overwrite=True,show_progress=False)
        img = imread(si['resourceIdentifier'], as_gray=True)
        img_resized = resize(img, dimension, anti_aliasing=True, mode='reflect')
        flat_data.append(img_resized.flatten()) 
        images.append(img_resized)
        target.append(df.loc[i].label[0])

    categories = list(categories)
    flat_data = np.array(flat_data)
    target = np.array(target)
    images = np.array(images)

    return Bunch(data=flat_data,
                 target=target,
                 target_names=categories,
                 images=images,
                 DESCR=descr)

# let user feed in 2 parameters, the dataset to mount or download, and the regularization rate of the logistic regression model
parser = argparse.ArgumentParser()
parser.add_argument('--regularization', type=float, dest='reg', default=0.5, help='regularization rate')
args = parser.parse_args()

# load train and test set
image_dataset = load_image_files()
# split
X_train, X_test, y_train, y_test = train_test_split(
    image_dataset.data, image_dataset.target, test_size=0.2,random_state=12)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape, sep = '\n')

# get hold of the current run
run = Run.get_context()

print('Train a logistic regression model with regularization rate of', args.reg)
clf = LogisticRegression(C=1.0/args.reg, solver="liblinear", multi_class="auto", random_state=12)
clf.fit(X_train, y_train)

print('Predict the test set')
y_pred = clf.predict(X_test)

print("Classification report for - \n{}:\n{}\n".format(
    clf, metrics.classification_report(y_test, y_pred)))

# calculate accuracy on the prediction
acc = np.average(y_pred == y_test)
print('Accuracy is', acc)

run.log('regularization rate', np.float(args.reg))
run.log('accuracy', np.float(acc))

os.makedirs('outputs', exist_ok=True)
# note file saved in the outputs folder is automatically uploaded into experiment record
joblib.dump(value=clf, filename='outputs/sklearn_ai_lab_defect_detection_model.pkl')