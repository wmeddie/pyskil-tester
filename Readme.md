# PySKIL Tester

This repository has everything you need to start a SKIL server, train a model, register it in SKIL's model metadata service, Deploy the model as a REST service and make predictions.

For more information on what SKIL has to offer see the [SKIL documentation](https://docs.skymind.ai).

## First step: Starting a SKIL server.

To start a temporary SKIL server install Docker.  If you are on Mac or Windows make sure the Docker VM is configured with 8GB of RAM or more.

Once your Docker is configured run the following command to start a temporary SKIL server.

    $ docker run --rm -it -p 9008:9008 -p 8080:8080 skymindops/skil-ce:1.1.4-2 bash /start-skil.sh

(On Windows with msys remove the `/start-skil.sh` from the command and run it in the bash prompt.)

## Second step: Training a model.

The next step is to train a model and save it to SKIL.  You can use your browser to train a model inside SKIL but you can also train the model outside of SKIL.  We will do that here.

First, create a python envrionment and install the requirements.txt with the following commands:

    $ virtualenv ./env
    $ source ./env/bin/activate
    $ pip install -r requirements.txt

Now train the model by running the `train_mnist.py` script:

    $ python train_mnist.py

This script will connect to SKIL and register an experiment to hold the models we train, and save a trained model.  It will output the ID of the model it registered in SKIL.

You can see the models and their accuracy using the SKIL UI by opening a browser to [http://localhost:9008/](http://localhost:9008) and browsing to the Workspace and experiment.

## Third step: Deploying a model.

You can deploy the last trained model by running the `eval_and_test.py` script.  It will deploy a model by ID or deploy a new model depending on the command line parameter you give it.  To deploy a model registered in SKIL specify the model's UUID like so:

    $ python eval_and_test.py --id 10378223-f528-11e8-b8fd-acde48001122

To deploy a specific model file run:

    $ python eval_and_test.py --file keras_mnist.h5

The script will deploy and start a model server, and then evaluate it's accuracy using the entire MNIST test set.
