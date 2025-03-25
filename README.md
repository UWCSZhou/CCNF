# Causally Consistent Normalizing Flow
This repo contains the code of the paper "Causally Consistent Normalizing Flow" published by AAAI25.

# Description
Welcome to the repository of [Causally Consistent Normalizing Flow
](https://arxiv.org/abs/2412.12401)! We appreciate your interest in our work and hope you find it valuable.

# Data in the paper
Under the **model** folder, you could find all the models and related results used in the paper.

Under the **dataset** folder, you could find all the dataset used in the paper.

# Training CCNF
To train a CCNF model, you first need to create venv in the root of the project:

    python -m venv ./env  
    source ./env/bin/activate
    pip install -r requirements.txt

Now you can train the CCNF model. For example, if I want to train a CCNF model on the network dataset with flow++, I would do:
    
    python -m train.train -model my -nf flow++ -d network

You could check the **train/arg_parser.py** for more options. 