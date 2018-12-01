#!/usr/bin/env python
import sys

import keras
from keras.datasets import mnist

import json

import numpy as np

import skil
from skil import Skil, WorkSpace, Experiment, Deployment, Model as SkilModel

import uuid

with open('.skil', 'r') as f:
    skil_conf = json.load(f)

# Connect to SKIL and create an experiment for storing our model experiments.
skil_server = Skil(
    host=skil_conf['host'],
    port=skil_conf['port'],
    user_id=skil_conf['username'], 
    password=skil_conf['password'])
work_space = skil.get_workspace_by_id(None, skil_server, skil_conf['workspace_id'])
experiment = skil.get_experiment_by_id(work_space, skil_conf['experiment_id'])

model = None
if sys.argv[1] == '--id':
    model = skil.get_model_by_id(experiment, sys.argv[2])
elif sys.argv[1] == '--file':
    model = SkilModel(sys.argv[2], name=sys.argv[2],  id=str(uuid.uuid1()), experiment=experiment)
else:
    print('Usage: python eval_and_test.py --id [model_id] | --file [path]')
    sys.exit(1)

deployment = None
if skil_conf['deployment_id'] == '':
    deployment = Deployment(skil_server, "mnist_models")
    skil_conf['deployment_id'] = deployment.id
    with open('.skil', 'w') as f:
        json.dump(skil_conf, f)
else:
    deployment = skil.get_deployement_by_id(skil_server, skil_conf['deployment_id'])

service = model.deploy(deployment=deployment)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_test = x_test.reshape(10000, 784)
x_test = x_test.astype('float32')
x_test /= 255

for i in xrange(10000):
    digit = x_test[i]
    label = y_test[i]
    result = service.predict_single(digit)
    predicted = np.argmax(result)
    actual = np.argmax(label)

    print("Predicted: %s Actual: %s Same: %s" % (predicted, actual, predicted == actual))
