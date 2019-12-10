# azureml-core of version 1.0.72 or higher is required
# azureml-contrib-dataset of version 1.0.72 or higher is required
from azureml.core import Workspace, Dataset, Datastore
import azureml.contrib.dataset
import azureml.dataprep.native

import os

subscription_id = os.environ['AZURE_SUBSCRIPTION_ID']
resource_group = 'ai-lab'
workspace_name = 'ailabml'

workspace = Workspace(subscription_id, resource_group, workspace_name)

# get dataset
ds = Dataset.get_by_name(workspace, name='Light Bulbs-2019-12-08 00:35:33')
df = ds.to_pandas_dataframe()

# download images
index = 0
datastore = None
while index < len(df):
    # image_url is a azureml.dataprep.native.StreamInfo object, convert to dic with to_pod()
    si = df.loc[index].image_url.to_pod()
    if index == 0:
        # retrieve datastore based on metadata from first row
        # assuming all images come from the same store
        # since they come from a single dataset
        datastore = Datastore.get(workspace, si['arguments']['datastoreName'])
    # download image locally
    datastore.download(target_path='.',prefix=si['resourceIdentifier'],overwrite=True,show_progress=True)
    index += 1

# create training, test sets
[training, test] = ds.random_split(0.8)

# build classification model based on image assets and labels...