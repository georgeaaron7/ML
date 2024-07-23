# /***This Challenge Lab is recommended for students who have enrolled in the Learning Tensorflow quest. You will be given a scenario and a set of tasks. Instead of following step-by-step instructions, you will use the skills learned from the labs in the quest to figure out how to complete the tasks on your own! An automated scoring system (shown on the Cloud Skills Boost lab instructions page) will provide feedback on whether you have completed your tasks correctly.

# When you take a Challenge Lab, you will not be taught Google Cloud concepts. To build the solution to the challenge presented, use skills learned from the labs in the Quest this challenge lab is part of. You are expected to extend your learned skills and complete all the TODO: comments in this notebook.

# Are you ready for the challenge?

# Scenario
# You were recently hired as a Machine Learning Engineer for an Optical Character Recognition app development team. Your manager has tasked you with building a machine learning model to recognize Hiragana alphabets. The challenge: your business requirements are that you have just 6 weeks to produce a model that achieves great than 90% accuracy to improve upon an existing bootstrapped solution. Furthermore, after doing some exploratory analysis in your startup's data warehouse, you found that you only have a small dataset of 60k images of alphabets to build a higher-performing solution.

# To build and deploy a high-performance machine learning model with limited data quickly, you will walk through training and deploying a CNN classifier for online predictions on Google Cloud's Vertex AI platform. Vertex AI is Google Cloud's next-generation machine learning development platform where you can leverage the latest ML pre-built components to significantly enhance your development productivity, scale your workflow and decision-making with your data, and accelerate time to value.

# Vertex AI: Challenge Lab

# First, you will progress through a typical experimentation workflow where you will build your custom CNN model script using tf.keras classification layers. You will then send the model code to a custom training job and run the custom training job using pre-built containers provided by Vertex AI to run training and prediction. Lastly, you will deploy the model to an endpoint so that you can use your model for predictions.

# Learning objectives
# Train a model on Vertex AI using the SDK for Python.
# Deploy a custom image classification model for online prediction using Vertex AI.
# Setup
# Installation***/
import os
import os
! pip3 install --user --upgrade google-cloud-aiplatform
! pip3 install --user --upgrade google-cloud-aiplatform
Requirement already satisfied: google-cloud-aiplatform in /opt/conda/lib/python3.10/site-packages (1.58.0)
! pip3 install --user --upgrade google-cloud-storage
! pip3 install --user --upgrade google-cloud-storage
Requirement already satisfied: google-cloud-storage in /opt/conda/lib/python3.10/site-packages (2.14.0)
Collecting google-cloud-storage
  Downloading google_cloud_storage-2.18.0-py2.py3-none-any.whl.metadata (9.1 kB)
Requirement already satisfied: google-auth<3.0dev,>=2.26.1 in /opt/conda/lib/python3.10/site-packages (from google-cloud-storage) (2.31.0)
Collecting google-api-core<3.0.0dev,>=2.15.0 (from google-cloud-storage)
  Downloading google_api_core-2.19.1-py3-none-any.whl.metadata (2.7 kB)
Requirement already satisfied: google-cloud-core<3.0dev,>=2.3.0 in /opt/conda/lib/python3.10/site-packages (from google-cloud-storage) (2.4.1)
Requirement already satisfied: google-resumable-media>=2.6.0 in /opt/conda/lib/python3.10/site-packages (from google-cloud-storage) (2.7.1)
Requirement already satisfied: requests<3.0.0dev,>=2.18.0 in /opt/conda/lib/python3.10/site-packages (from google-cloud-storage) (2.32.3)
Requirement already satisfied: google-crc32c<2.0dev,>=1.0 in /opt/conda/lib/python3.10/site-packages (from google-cloud-storage) (1.5.0)
Requirement already satisfied: googleapis-common-protos<2.0.dev0,>=1.56.2 in /opt/conda/lib/python3.10/site-packages (from google-api-core<3.0.0dev,>=2.15.0->google-cloud-storage) (1.63.1)
Requirement already satisfied: protobuf!=3.20.0,!=3.20.1,!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<6.0.0.dev0,>=3.19.5 in /opt/conda/lib/python3.10/site-packages (from google-api-core<3.0.0dev,>=2.15.0->google-cloud-storage) (3.19.6)
Requirement already satisfied: proto-plus<2.0.0dev,>=1.22.3 in /opt/conda/lib/python3.10/site-packages (from google-api-core<3.0.0dev,>=2.15.0->google-cloud-storage) (1.24.0)
Requirement already satisfied: cachetools<6.0,>=2.0.0 in /opt/conda/lib/python3.10/site-packages (from google-auth<3.0dev,>=2.26.1->google-cloud-storage) (4.2.4)
Requirement already satisfied: pyasn1-modules>=0.2.1 in /opt/conda/lib/python3.10/site-packages (from google-auth<3.0dev,>=2.26.1->google-cloud-storage) (0.4.0)
Requirement already satisfied: rsa<5,>=3.1.4 in /opt/conda/lib/python3.10/site-packages (from google-auth<3.0dev,>=2.26.1->google-cloud-storage) (4.9)
Requirement already satisfied: charset-normalizer<4,>=2 in /opt/conda/lib/python3.10/site-packages (from requests<3.0.0dev,>=2.18.0->google-cloud-storage) (3.3.2)
Requirement already satisfied: idna<4,>=2.5 in /opt/conda/lib/python3.10/site-packages (from requests<3.0.0dev,>=2.18.0->google-cloud-storage) (3.7)
Requirement already satisfied: urllib3<3,>=1.21.1 in /opt/conda/lib/python3.10/site-packages (from requests<3.0.0dev,>=2.18.0->google-cloud-storage) (1.26.19)
Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/lib/python3.10/site-packages (from requests<3.0.0dev,>=2.18.0->google-cloud-storage) (2024.7.4)
Requirement already satisfied: pyasn1<0.7.0,>=0.4.6 in /opt/conda/lib/python3.10/site-packages (from pyasn1-modules>=0.2.1->google-auth<3.0dev,>=2.26.1->google-cloud-storage) (0.6.0)
Downloading google_cloud_storage-2.18.0-py2.py3-none-any.whl (130 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 130.5/130.5 kB 7.2 MB/s eta 0:00:00
Downloading google_api_core-2.19.1-py3-none-any.whl (139 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 139.4/139.4 kB 14.8 MB/s eta 0:00:00
Installing collected packages: google-api-core, google-cloud-storage
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
google-api-python-client 1.8.0 requires google-api-core<2dev,>=1.13.0, but you have google-api-core 2.19.1 which is incompatible.
google-cloud-pubsub 2.21.4 requires grpcio<2.0dev,>=1.51.3, but you have grpcio 1.48.0 which is incompatible.
Successfully installed google-api-core-2.19.1 google-cloud-storage-2.18.0
Install the latest version of google-cloud-logging library.

! pip3 install --user --upgrade google-cloud-logging
! pip3 install --user --upgrade google-cloud-logging
Collecting google-cloud-logging
  Downloading google_cloud_logging-3.10.0-py2.py3-none-any.whl.metadata (4.9 kB)
Requirement already satisfied: google-api-core!=2.0.*,!=2.1.*,!=2.10.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,!=2.8.*,!=2.9.*,<3.0.0dev,>=1.34.1 in /home/jupyter/.local/lib/python3.10/site-packages (from google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.10.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,!=2.8.*,!=2.9.*,<3.0.0dev,>=1.34.1->google-cloud-logging) (2.19.1)
Requirement already satisfied: google-auth!=2.24.0,!=2.25.0,<3.0.0dev,>=2.14.1 in /opt/conda/lib/python3.10/site-packages (from google-cloud-logging) (2.31.0)
Collecting google-cloud-appengine-logging<2.0.0dev,>=0.1.0 (from google-cloud-logging)
  Downloading google_cloud_appengine_logging-1.4.4-py2.py3-none-any.whl.metadata (5.4 kB)
Collecting google-cloud-audit-log<1.0.0dev,>=0.1.0 (from google-cloud-logging)
  Downloading google_cloud_audit_log-0.2.5-py2.py3-none-any.whl.metadata (1.4 kB)
Requirement already satisfied: google-cloud-core<3.0.0dev,>=2.0.0 in /opt/conda/lib/python3.10/site-packages (from google-cloud-logging) (2.4.1)
Requirement already satisfied: grpc-google-iam-v1<1.0.0dev,>=0.12.4 in /opt/conda/lib/python3.10/site-packages (from google-cloud-logging) (0.12.7)
Requirement already satisfied: proto-plus<2.0.0dev,>=1.22.0 in /opt/conda/lib/python3.10/site-packages (from google-cloud-logging) (1.24.0)
Requirement already satisfied: protobuf!=3.20.0,!=3.20.1,!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<5.0.0dev,>=3.19.5 in /opt/conda/lib/python3.10/site-packages (from google-cloud-logging) (3.19.6)
Requirement already satisfied: googleapis-common-protos<2.0.dev0,>=1.56.2 in /opt/conda/lib/python3.10/site-packages (from google-api-core!=2.0.*,!=2.1.*,!=2.10.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,!=2.8.*,!=2.9.*,<3.0.0dev,>=1.34.1->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.10.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,!=2.8.*,!=2.9.*,<3.0.0dev,>=1.34.1->google-cloud-logging) (1.63.1)
Requirement already satisfied: requests<3.0.0.dev0,>=2.18.0 in /opt/conda/lib/python3.10/site-packages (from google-api-core!=2.0.*,!=2.1.*,!=2.10.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,!=2.8.*,!=2.9.*,<3.0.0dev,>=1.34.1->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.10.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,!=2.8.*,!=2.9.*,<3.0.0dev,>=1.34.1->google-cloud-logging) (2.32.3)
Requirement already satisfied: grpcio<2.0dev,>=1.33.2 in /opt/conda/lib/python3.10/site-packages (from google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.10.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,!=2.8.*,!=2.9.*,<3.0.0dev,>=1.34.1->google-cloud-logging) (1.48.0)
Requirement already satisfied: grpcio-status<2.0.dev0,>=1.33.2 in /opt/conda/lib/python3.10/site-packages (from google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.10.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,!=2.8.*,!=2.9.*,<3.0.0dev,>=1.34.1->google-cloud-logging) (1.48.0)
Requirement already satisfied: cachetools<6.0,>=2.0.0 in /opt/conda/lib/python3.10/site-packages (from google-auth!=2.24.0,!=2.25.0,<3.0.0dev,>=2.14.1->google-cloud-logging) (4.2.4)
Requirement already satisfied: pyasn1-modules>=0.2.1 in /opt/conda/lib/python3.10/site-packages (from google-auth!=2.24.0,!=2.25.0,<3.0.0dev,>=2.14.1->google-cloud-logging) (0.4.0)
Requirement already satisfied: rsa<5,>=3.1.4 in /opt/conda/lib/python3.10/site-packages (from google-auth!=2.24.0,!=2.25.0,<3.0.0dev,>=2.14.1->google-cloud-logging) (4.9)
Collecting protobuf!=3.20.0,!=3.20.1,!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<5.0.0dev,>=3.19.5 (from google-cloud-logging)
  Downloading protobuf-4.25.3-cp37-abi3-manylinux2014_x86_64.whl.metadata (541 bytes)
Requirement already satisfied: six>=1.5.2 in /opt/conda/lib/python3.10/site-packages (from grpcio<2.0dev,>=1.33.2->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.10.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,!=2.8.*,!=2.9.*,<3.0.0dev,>=1.34.1->google-cloud-logging) (1.16.0)
Requirement already satisfied: pyasn1<0.7.0,>=0.4.6 in /opt/conda/lib/python3.10/site-packages (from pyasn1-modules>=0.2.1->google-auth!=2.24.0,!=2.25.0,<3.0.0dev,>=2.14.1->google-cloud-logging) (0.6.0)
Requirement already satisfied: charset-normalizer<4,>=2 in /opt/conda/lib/python3.10/site-packages (from requests<3.0.0.dev0,>=2.18.0->google-api-core!=2.0.*,!=2.1.*,!=2.10.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,!=2.8.*,!=2.9.*,<3.0.0dev,>=1.34.1->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.10.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,!=2.8.*,!=2.9.*,<3.0.0dev,>=1.34.1->google-cloud-logging) (3.3.2)
Requirement already satisfied: idna<4,>=2.5 in /opt/conda/lib/python3.10/site-packages (from requests<3.0.0.dev0,>=2.18.0->google-api-core!=2.0.*,!=2.1.*,!=2.10.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,!=2.8.*,!=2.9.*,<3.0.0dev,>=1.34.1->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.10.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,!=2.8.*,!=2.9.*,<3.0.0dev,>=1.34.1->google-cloud-logging) (3.7)
Requirement already satisfied: urllib3<3,>=1.21.1 in /opt/conda/lib/python3.10/site-packages (from requests<3.0.0.dev0,>=2.18.0->google-api-core!=2.0.*,!=2.1.*,!=2.10.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,!=2.8.*,!=2.9.*,<3.0.0dev,>=1.34.1->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.10.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,!=2.8.*,!=2.9.*,<3.0.0dev,>=1.34.1->google-cloud-logging) (1.26.19)
Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/lib/python3.10/site-packages (from requests<3.0.0.dev0,>=2.18.0->google-api-core!=2.0.*,!=2.1.*,!=2.10.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,!=2.8.*,!=2.9.*,<3.0.0dev,>=1.34.1->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.10.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,!=2.8.*,!=2.9.*,<3.0.0dev,>=1.34.1->google-cloud-logging) (2024.7.4)
Downloading google_cloud_logging-3.10.0-py2.py3-none-any.whl (213 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 213.4/213.4 kB 10.8 MB/s eta 0:00:00
Downloading google_cloud_appengine_logging-1.4.4-py2.py3-none-any.whl (15 kB)
Downloading google_cloud_audit_log-0.2.5-py2.py3-none-any.whl (12 kB)
Downloading protobuf-4.25.3-cp37-abi3-manylinux2014_x86_64.whl (294 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 294.6/294.6 kB 26.4 MB/s eta 0:00:00
Installing collected packages: protobuf, google-cloud-audit-log, google-cloud-appengine-logging, google-cloud-logging
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
apache-beam 2.46.0 requires grpcio!=1.48.0,<2,>=1.33.1, but you have grpcio 1.48.0 which is incompatible.
apache-beam 2.46.0 requires protobuf<4,>3.12.2, but you have protobuf 4.25.3 which is incompatible.
google-api-python-client 1.8.0 requires google-api-core<2dev,>=1.13.0, but you have google-api-core 2.19.1 which is incompatible.
google-cloud-bigtable 1.7.3 requires protobuf<4.0.0dev, but you have protobuf 4.25.3 which is incompatible.
google-cloud-datastore 1.15.5 requires protobuf<4.0.0dev, but you have protobuf 4.25.3 which is incompatible.
google-cloud-language 1.3.2 requires protobuf<4.0.0dev, but you have protobuf 4.25.3 which is incompatible.
google-cloud-pubsub 2.21.4 requires grpcio<2.0dev,>=1.51.3, but you have grpcio 1.48.0 which is incompatible.
google-cloud-videointelligence 1.16.3 requires protobuf<4.0.0dev, but you have protobuf 4.25.3 which is incompatible.
kfp 2.5.0 requires protobuf<4,>=3.13.0, but you have protobuf 4.25.3 which is incompatible.
kfp-pipeline-spec 0.2.2 requires protobuf<4,>=3.13.0, but you have protobuf 4.25.3 which is incompatible.
tensorboard 2.11.2 requires protobuf<4,>=3.9.2, but you have protobuf 4.25.3 which is incompatible.
tensorboardx 2.6 requires protobuf<4,>=3.8.0, but you have protobuf 4.25.3 which is incompatible.
tensorflow 2.11.0 requires protobuf<3.20,>=3.9.2, but you have protobuf 4.25.3 which is incompatible.
tensorflow-metadata 0.14.0 requires protobuf<4,>=3.7, but you have protobuf 4.25.3 which is incompatible.
tensorflow-serving-api 2.11.0 requires protobuf<3.20,>=3.9.2, but you have protobuf 4.25.3 which is incompatible.
tensorflow-transform 0.14.0 requires protobuf<4,>=3.7, but you have protobuf 4.25.3 which is incompatible.
Successfully installed google-cloud-appengine-logging-1.4.4 google-cloud-audit-log-0.2.5 google-cloud-logging-3.10.0 protobuf-4.25.3
Downgrade protobuf for tensorflow datasets compatibility

! pip3 install --user protobuf==3.19.*
! pip3 install --user protobuf==3.19.*
Collecting protobuf==3.19.*
  Downloading protobuf-3.19.6-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (787 bytes)
Downloading protobuf-3.19.6-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.1 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.1/1.1 MB 32.9 MB/s eta 0:00:00
Installing collected packages: protobuf
  Attempting uninstall: protobuf
    Found existing installation: protobuf 4.25.3
    Uninstalling protobuf-4.25.3:
      Successfully uninstalled protobuf-4.25.3
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
google-cloud-appengine-logging 1.4.4 requires protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<6.0.0dev,>=3.20.2, but you have protobuf 3.19.6 which is incompatible.
apache-beam 2.46.0 requires grpcio!=1.48.0,<2,>=1.33.1, but you have grpcio 1.48.0 which is incompatible.
google-api-python-client 1.8.0 requires google-api-core<2dev,>=1.13.0, but you have google-api-core 2.19.1 which is incompatible.
google-cloud-pubsub 2.21.4 requires grpcio<2.0dev,>=1.51.3, but you have grpcio 1.48.0 which is incompatible.
Successfully installed protobuf-3.19.6
Install the pillow library for loading images.

! pip3 install --user --upgrade pillow
! pip3 install --user --upgrade pillow
Requirement already satisfied: pillow in /opt/conda/lib/python3.10/site-packages (10.4.0)
Install the numpy library for manipulation of image data.

! pip3 install numpy==1.24.4 --user
! pip3 install numpy==1.24.4 --user
Requirement already satisfied: numpy==1.24.4 in /opt/conda/lib/python3.10/site-packages (1.24.4)
You can safely ignore errors during the numpy installation.

Restart the kernel
Once you've installed everything, you need to restart the notebook kernel so it can find the packages.

import os

if not os.getenv("IS_TESTING"):
    # Automatically restart kernel after installs
    import IPython

    app = IPython.Application.instance()
    app.kernel.do_shutdown(True)
import os
​
if not os.getenv("IS_TESTING"):
    # Automatically restart kernel after installs
    import IPython
​
    app = IPython.Application.instance()
    app.kernel.do_shutdown(True)
Set your project ID
If you don't know your project ID, you may be able to get your project ID using gcloud.

import os

# Retrieve and set PROJECT_ID environment variables.
# TODO: fill in PROJECT_ID.

if not os.getenv("IS_TESTING"):
    # Get your Google Cloud project ID from gcloud
    PROJECT_ID = "qwiklabs-gcp-02-836dceb82b49"
import os
​
# Retrieve and set PROJECT_ID environment variables.
# TODO: fill in PROJECT_ID.
​
if not os.getenv("IS_TESTING"):
    # Get your Google Cloud project ID from gcloud
    PROJECT_ID = "qwiklabs-gcp-02-836dceb82b49"
PROJECT_ID
'qwiklabs-gcp-02-836dceb82b49'
Timestamp
If you are in a live tutorial session, you might be using a shared test account or project. To avoid name collisions between users on resources created, you create a timestamp for each instance session, and append it onto the name of resources you create in this tutorial.

from datetime import datetime
​
TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")
Create a Cloud Storage bucket
The following steps are required, regardless of your notebook environment.

When you submit a training job using the Cloud SDK, you upload a Python package containing your training code to a Cloud Storage bucket. Vertex AI runs the code from this package. In this tutorial, Vertex AI also saves the trained model that results from your job in the same bucket. Using this model artifact, you can then create Vertex AI model and endpoint resources in order to serve online predictions.

Set the name of your Cloud Storage bucket below. It must be unique across all Cloud Storage buckets.

Note: Replace the REGION with the associated region mentioned in the qwiklabs resource panel.

REGION = "europe-west4"  # @param {type:"string"}
REGION = "europe-west4"  # @param {type:"string"}
# TODO: Create a globally unique Google Cloud Storage bucket name for artifact storage.
# HINT: Start the name with gs://
BUCKET_NAME = "gs://qwiklabs-gcp-02-836dceb82b49"
# TODO: Create a globally unique Google Cloud Storage bucket name for artifact storage.
# HINT: Start the name with gs://
BUCKET_NAME = "gs://qwiklabs-gcp-02-836dceb82b49"
! gsutil mb -l $REGION $BUCKET_NAME
! gsutil mb -l $REGION $BUCKET_NAME
Creating gs://qwiklabs-gcp-02-836dceb82b49/...
Set up variables
Next, set up some variables used throughout the tutorial.

Import Vertex SDK for Python
Import the Vertex SDK for Python into your Python environment and initialize it.

import os
import sys

from google.cloud import aiplatform
from google.cloud.aiplatform import gapic as aip

aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET_NAME)
import os
import sys
​
from google.cloud import aiplatform
from google.cloud.aiplatform import gapic as aip
​
aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET_NAME)
Set hardware accelerators
Here to run a container image on a CPU, we set the variables TRAIN_GPU/TRAIN_NGPU and DEPLOY_GPU/DEPLOY_NGPU to (None, None) since this notebook is meant to be run in a Qwiklab environment where GPUs cannot be provisioned.

Note: If you happen to be running this notebook from your personal GCP account, set the variables TRAIN_GPU/TRAIN_NGPU and DEPLOY_GPU/DEPLOY_NGPU to use a container image supporting a GPU and the number of GPUs allocated to the virtual machine (VM) instance. For example, to use a GPU container image with 4 Nvidia Tesla K80 GPUs allocated to each VM, you would specify:

(aip.AcceleratorType.NVIDIA_TESLA_K80, 4)
See the locations where accelerators are available.

TRAIN_GPU, TRAIN_NGPU = (None, None)
DEPLOY_GPU, DEPLOY_NGPU = (None, None)
TRAIN_GPU, TRAIN_NGPU = (None, None)
DEPLOY_GPU, DEPLOY_NGPU = (None, None)
Set pre-built containers
There are two ways you can train a custom model using a container image:

Use a Google Cloud prebuilt container. If you use a prebuilt container, you will additionally specify a Python package to install into the container image. This Python package contains your code for training a custom model.

Use your own custom container image. If you use your own container, the container needs to contain your code for training a custom model.

Here you will use pre-built containers provided by Vertex AI to run training and prediction.

For the latest list, see Pre-built containers for training and Pre-built containers for prediction

TRAIN_VERSION = "tf-cpu.2-8"
DEPLOY_VERSION = "tf2-cpu.2-8"

TRAIN_IMAGE = "us-docker.pkg.dev/vertex-ai/training/{}:latest".format(TRAIN_VERSION)
DEPLOY_IMAGE = "us-docker.pkg.dev/vertex-ai/prediction/{}:latest".format(DEPLOY_VERSION)

print("Training:", TRAIN_IMAGE, TRAIN_GPU, TRAIN_NGPU)
print("Deployment:", DEPLOY_IMAGE, DEPLOY_GPU, DEPLOY_NGPU)
TRAIN_VERSION = "tf-cpu.2-8"
DEPLOY_VERSION = "tf2-cpu.2-8"
​
TRAIN_IMAGE = "us-docker.pkg.dev/vertex-ai/training/{}:latest".format(TRAIN_VERSION)
DEPLOY_IMAGE = "us-docker.pkg.dev/vertex-ai/prediction/{}:latest".format(DEPLOY_VERSION)
​
print("Training:", TRAIN_IMAGE, TRAIN_GPU, TRAIN_NGPU)
print("Deployment:", DEPLOY_IMAGE, DEPLOY_GPU, DEPLOY_NGPU)
Training: us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-8:latest None None
Deployment: us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-8:latest None None
Set machine types
Next, set the machine types to use for training and prediction.

Set the variables TRAIN_COMPUTE and DEPLOY_COMPUTE to configure your compute resources for training and prediction.
machine type
n1-standard: 3.75GB of memory per vCPU
n1-highmem: 6.5GB of memory per vCPU
n1-highcpu: 0.9 GB of memory per vCPU
vCPUs: number of [2, 4, 8, 16, 32, 64, 96 ]
Note: The following is not supported for training:

standard: 2 vCPUs
highcpu: 2, 4 and 8 vCPUs
Note: You may also use n2 and e2 machine types for training and deployment, but they do not support GPUs.

MACHINE_TYPE = "n1-standard"

VCPU = "4"
TRAIN_COMPUTE = MACHINE_TYPE + "-" + VCPU
print("Train machine type", TRAIN_COMPUTE)

MACHINE_TYPE = "n1-standard"

VCPU = "4"
DEPLOY_COMPUTE = MACHINE_TYPE + "-" + VCPU
print("Deploy machine type", DEPLOY_COMPUTE)
MACHINE_TYPE = "n1-standard"
​
VCPU = "4"
TRAIN_COMPUTE = MACHINE_TYPE + "-" + VCPU
print("Train machine type", TRAIN_COMPUTE)
​
MACHINE_TYPE = "n1-standard"
​
VCPU = "4"
DEPLOY_COMPUTE = MACHINE_TYPE + "-" + VCPU
print("Deploy machine type", DEPLOY_COMPUTE)
Train machine type n1-standard-4
Deploy machine type n1-standard-4
Training script
In the next cell, you will write the contents of the training script, task.py.

%%writefile task.py
# Training kmnist using CNN

import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.python.client import device_lib
import argparse
import os
import sys
tfds.disable_progress_bar()

parser = argparse.ArgumentParser()

parser.add_argument('--epochs', dest='epochs',
                    default=10, type=int,
                    help='Number of epochs.')

args = parser.parse_args()

print('Python Version = {}'.format(sys.version))
print('TensorFlow Version = {}'.format(tf.__version__))
print('TF_CONFIG = {}'.format(os.environ.get('TF_CONFIG', 'Not found')))
print('DEVICES', device_lib.list_local_devices())

# Define batch size
BATCH_SIZE = 32

# Load the dataset
datasets, info = tfds.load('kmnist', with_info=True, as_supervised=True)

# Normalize and batch process the dataset
ds_train = datasets['train'].map(lambda x, y: (tf.cast(x, tf.float32)/255.0, y)).batch(BATCH_SIZE)


# Build the Convolutional Neural Network
model = tf.keras.models.Sequential([                               
      tf.keras.layers.Conv2D(16, (3,3), activation=tf.nn.relu, input_shape=(28, 28, 1), padding = "same"),
      tf.keras.layers.MaxPooling2D(2,2),
      tf.keras.layers.Conv2D(16, (3,3), activation=tf.nn.relu, padding = "same"),
      tf.keras.layers.MaxPooling2D(2,2),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128, activation=tf.nn.relu),
      # TODO: Write the last layer.
      # Hint: KMNIST has 10 output classes.
    
    tf.keras.layers.Dense(10, activation=tf.nn.relu)
      
    ])

model.compile(optimizer = tf.keras.optimizers.Adam(),
      loss = tf.keras.losses.SparseCategoricalCrossentropy(),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])



# Train and save the model

MODEL_DIR = os.getenv("AIP_MODEL_DIR")

model.fit(ds_train, epochs=args.epochs)

# TODO: Save your CNN classifier. 
# Hint: Save it to MODEL_DIR.
model.save(MODEL_DIR)
%%writefile task.py
# Training kmnist using CNN
​
import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.python.client import device_lib
import argparse
import os
import sys
tfds.disable_progress_bar()
​
parser = argparse.ArgumentParser()
​
parser.add_argument('--epochs', dest='epochs',
                    default=10, type=int,
                    help='Number of epochs.')
​
args = parser.parse_args()
​
print('Python Version = {}'.format(sys.version))
print('TensorFlow Version = {}'.format(tf.__version__))
print('TF_CONFIG = {}'.format(os.environ.get('TF_CONFIG', 'Not found')))
print('DEVICES', device_lib.list_local_devices())
​
# Define batch size
BATCH_SIZE = 32
​
# Load the dataset
datasets, info = tfds.load('kmnist', with_info=True, as_supervised=True)
​
# Normalize and batch process the dataset
ds_train = datasets['train'].map(lambda x, y: (tf.cast(x, tf.float32)/255.0, y)).batch(BATCH_SIZE)
​
​
# Build the Convolutional Neural Network
model = tf.keras.models.Sequential([                               
      tf.keras.layers.Conv2D(16, (3,3), activation=tf.nn.relu, input_shape=(28, 28, 1), padding = "same"),
      tf.keras.layers.MaxPooling2D(2,2),
      tf.keras.layers.Conv2D(16, (3,3), activation=tf.nn.relu, padding = "same"),
      tf.keras.layers.MaxPooling2D(2,2),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128, activation=tf.nn.relu),
      # TODO: Write the last layer.
      # Hint: KMNIST has 10 output classes.
    
    tf.keras.layers.Dense(10, activation=tf.nn.relu)
      
    ])
​
model.compile(optimizer = tf.keras.optimizers.Adam(),
      loss = tf.keras.losses.SparseCategoricalCrossentropy(),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
​
​
​
# Train and save the model
​
MODEL_DIR = os.getenv("AIP_MODEL_DIR")
​
model.fit(ds_train, epochs=args.epochs)
​
# TODO: Save your CNN classifier. 
# Hint: Save it to MODEL_DIR.
model.save(MODEL_DIR)
Overwriting task.py
Define the command args for the training script
Prepare the command-line arguments to pass to your training script.

args: The command line arguments to pass to the corresponding Python module. In this example, they will be:
"--epochs=" + EPOCHS: The number of epochs for training.
JOB_NAME = "custom_job_" + TIMESTAMP
MODEL_DIR = "{}/{}".format(BUCKET_NAME, JOB_NAME)

EPOCHS = 5

CMDARGS = [
    "--epochs=" + str(EPOCHS),
]
JOB_NAME = "custom_job_" + TIMESTAMP
MODEL_DIR = "{}/{}".format(BUCKET_NAME, JOB_NAME)
​
EPOCHS = 5
​
CMDARGS = [
    "--epochs=" + str(EPOCHS),
]
Train the model
Define your custom training job on Vertex AI.

job = aiplatform.CustomTrainingJob(
    display_name=JOB_NAME,
    requirements=["tensorflow_datasets==4.6.0"],
    # TODO: fill in the remaining arguments for the CustomTrainingJob function.
    script_path="task.py",
    container_uri=TRAIN_IMAGE,
    model_serving_container_image_uri = DEPLOY_IMAGE,
)

MODEL_DISPLAY_NAME = "kmnist-" + TIMESTAMP

# Start the training
model = job.run(
    model_display_name=MODEL_DISPLAY_NAME,
    replica_count=1,
    accelerator_count=0,
    # TODO: fill in the remaining arguments to run the custom training job function.
    args=CMDARGS,
    machine_type=TRAIN_COMPUTE,
)
job = aiplatform.CustomTrainingJob(
    display_name=JOB_NAME,
    requirements=["tensorflow_datasets==4.6.0"],
    # TODO: fill in the remaining arguments for the CustomTrainingJob function.
    script_path="task.py",
    container_uri=TRAIN_IMAGE,
    model_serving_container_image_uri = DEPLOY_IMAGE,
)
​
MODEL_DISPLAY_NAME = "kmnist-" + TIMESTAMP
​
# Start the training
model = job.run(
    model_display_name=MODEL_DISPLAY_NAME,
    replica_count=1,
    accelerator_count=0,
    # TODO: fill in the remaining arguments to run the custom training job function.
    args=CMDARGS,
    machine_type=TRAIN_COMPUTE,
)
Training script copied to:
gs://qwiklabs-gcp-02-836dceb82b49/aiplatform-2024-07-23-22:27:11.291-aiplatform_custom_trainer_script-0.1.tar.gz.
Training Output directory:
gs://qwiklabs-gcp-02-836dceb82b49/aiplatform-custom-training-2024-07-23-22:27:11.382 
View Training:
https://console.cloud.google.com/ai/platform/locations/europe-west4/training/489204334156840960?project=132014450357
CustomTrainingJob projects/132014450357/locations/europe-west4/trainingPipelines/489204334156840960 current state:
PipelineState.PIPELINE_STATE_PENDING
CustomTrainingJob projects/132014450357/locations/europe-west4/trainingPipelines/489204334156840960 current state:
PipelineState.PIPELINE_STATE_PENDING
CustomTrainingJob projects/132014450357/locations/europe-west4/trainingPipelines/489204334156840960 current state:
PipelineState.PIPELINE_STATE_PENDING
View backing custom job:
https://console.cloud.google.com/ai/platform/locations/europe-west4/training/957295571159220224?project=132014450357
CustomTrainingJob projects/132014450357/locations/europe-west4/trainingPipelines/489204334156840960 current state:
PipelineState.PIPELINE_STATE_RUNNING
CustomTrainingJob projects/132014450357/locations/europe-west4/trainingPipelines/489204334156840960 current state:
PipelineState.PIPELINE_STATE_RUNNING
CustomTrainingJob projects/132014450357/locations/europe-west4/trainingPipelines/489204334156840960 current state:
PipelineState.PIPELINE_STATE_RUNNING
CustomTrainingJob run completed. Resource name: projects/132014450357/locations/europe-west4/trainingPipelines/489204334156840960
Model available at projects/132014450357/locations/europe-west4/models/2928452463558131712
Deploy the model
Before you use your model to make predictions, you need to deploy it to an Endpoint. You can do this by calling the deploy function on the Model resource.

DEPLOYED_NAME = "kmnist_deployed-" + TIMESTAMP

TRAFFIC_SPLIT = {"0": 100}

MIN_NODES = 1
MAX_NODES = 1

endpoint = model.deploy(
        deployed_model_display_name=DEPLOYED_NAME,
        accelerator_type=None,
        accelerator_count=0,
        # TODO: fill in the remaining arguments to deploy the model to an endpoint.
    traffic_split=TRAFFIC_SPLIT,
    machine_type=DEPLOY_COMPUTE,
    min_replica_count=MIN_NODES,
    max_replica_count=MAX_NODES,
    )
DEPLOYED_NAME = "kmnist_deployed-" + TIMESTAMP
​
TRAFFIC_SPLIT = {"0": 100}
​
MIN_NODES = 1
MAX_NODES = 1
​
endpoint = model.deploy(
        deployed_model_display_name=DEPLOYED_NAME,
        accelerator_type=None,
        accelerator_count=0,
        # TODO: fill in the remaining arguments to deploy the model to an endpoint.
    traffic_split=TRAFFIC_SPLIT,
    machine_type=DEPLOY_COMPUTE,
    min_replica_count=MIN_NODES,
    max_replica_count=MAX_NODES,
    )
Creating Endpoint
Create Endpoint backing LRO: projects/132014450357/locations/europe-west4/endpoints/4938456476155904000/operations/1621817858998665216
Endpoint created. Resource name: projects/132014450357/locations/europe-west4/endpoints/4938456476155904000
To use this Endpoint in another session:
endpoint = aiplatform.Endpoint('projects/132014450357/locations/europe-west4/endpoints/4938456476155904000')
Deploying model to Endpoint : projects/132014450357/locations/europe-west4/endpoints/4938456476155904000
Deploy Endpoint model backing LRO: projects/132014450357/locations/europe-west4/endpoints/4938456476155904000/operations/6158350058644307968
Endpoint model deployed. Resource name: projects/132014450357/locations/europe-west4/endpoints/4938456476155904000
Make an online prediction request
Send an online prediction request to your deployed model.

import tensorflow_datasets as tfds
import numpy as np

tfds.disable_progress_bar()
import tensorflow_datasets as tfds
import numpy as np
​
tfds.disable_progress_bar()
datasets, info = tfds.load('kmnist', batch_size=-1, with_info=True, as_supervised=True)

test_dataset = datasets['test']
datasets, info = tfds.load('kmnist', batch_size=-1, with_info=True, as_supervised=True)
​
test_dataset = datasets['test']
x_test, y_test = tfds.as_numpy(test_dataset)

# Normalize (rescale) the pixel data by dividing each pixel by 255. 
x_test = x_test.astype('float32') / 255.
x_test.shape, y_test.shape
x_test, y_test = tfds.as_numpy(test_dataset)
​
# Normalize (rescale) the pixel data by dividing each pixel by 255. 
x_test = x_test.astype('float32') / 255.
x_test.shape, y_test.shape
((10000, 28, 28, 1), (10000,))
#@title Pick the number of test images
NUM_TEST_IMAGES = 20 #@param {type:"slider", min:1, max:20, step:1}
x_test, y_test = x_test[:NUM_TEST_IMAGES], y_test[:NUM_TEST_IMAGES]
#@title Pick the number of test images
NUM_TEST_IMAGES = 20 #@param {type:"slider", min:1, max:20, step:1}
x_test, y_test = x_test[:NUM_TEST_IMAGES], y_test[:NUM_TEST_IMAGES]
Send the prediction request
Logging module added to log the prediction result

# Import and configure logging
from google.cloud import logging
logging_client = logging.Client()
logger = logging_client.logger('challenge-notebook')
# Import and configure logging
from google.cloud import logging
logging_client = logging.Client()
logger = logging_client.logger('challenge-notebook')
Now that you have test images, you can use them to send a prediction request.

# TODO: use your Endpoint to return prediction for your x_test.
predictions = endpoint.predict(instances=x_test.tolist())

# TODO: use your Endpoint to return prediction for your x_test.
predictions = endpoint.predict(instances=x_test.tolist())
​
y_predicted = np.argmax(predictions.predictions, axis=1)
​
correct = sum(y_predicted == np.array(y_test.tolist()))
total = len(y_predicted)
​
logger.log_text(str(correct/total))
​
print(
    f"Correct predictions = {correct}, Total predictions = {total}, Accuracy = {correct/total}"
)
Correct predictions = 3, Total predictions = 20, Accuracy = 0.15
​
