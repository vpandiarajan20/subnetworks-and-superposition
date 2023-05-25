# Subnetworks and Superposition

This research attempts to identify polysemantic neurons in AlexNet and then employ continuous sparsification (i.e. L0 regularization) to create subnetworks such that each polysemantic neuron is monosemantic. We aim to show that these polysemantic neurons arose by independent statistical process (by combinatorial analysis of subnetwork masks). Our work was heavily inspired by Hamblin et. al's work on Circuit Exploration, and uses it as a baseline.

## Setup

Before running code, download the circuit explorer in the directory you plan to use and read the README.
https://github.com/chrishamblin7/circuit_explorer/tree/main. Much of our code is dependent on functions created by Hamblin et. al.

Then, follow the steps below to run our experiments.

* `python3 -m venv env`
* `source env/bin/activate`

Install dependencies: `pip install -r requirements.txt`

Add the circuit_explorer package itself: `pip install -e .`


## Important Files

* Run `HDBScan_Clustering.py` to perform clustering to detect polysemantic neurons and generate graphs of the two clusters (code has not been fully tested). Pass in parameters (can be found in file at bottom) to change hyperparameters

* Run `Subnetwork_Training.py` with passed hyperparameters to perform continuous sparsification to find Subnetworks and Save them

* `models/Masked_AlexNet.py` contains code that has masked alexnet for continuous sparsification available


* `Detecting_and_Subnetworking_Polysemantic_Neurons.ipynb` : Notebook with Majority of the Code for Generating Visuals (also has code to cluster and run ActGrad, SNIP, and Force)

## Additional Notes
In order to generate visualizations run the `Detecting_and_Subnetworking_Polysemantic_Neurons` notebook.
Run the Subnetwork Training file in order to train the masked network. Specify parameters such as layer, unit, batch size, temperature, epochs, learning rate, schedule, decay, lambda normalization constant, temperature, mask initial value, and min cluster size. Use the switch cluster labels parameter to train the two individual subnetworks. For more information, please read our [paper](https://github.com/surajK610/subnetworks-and-superposition/blob/main/Subnetworks_and_Superposition.pdf).