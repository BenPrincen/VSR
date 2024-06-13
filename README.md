The initial code for SRGAN was forked from [Aladdin Persson's Machine Learning collection](https://github.com/aladdinpersson/Machine-Learning-Collection)

# Training details

Make sure the RealVSR training dataset is located at `datasets/RealVSR/train/`. The dataset class assumes that 
the training dataset already has GT images located in `datasets/RealVSR/train/GT` and low quality images 
located in `datasets/RealVSR/train/LQ`. The format of the dataset should mimic RealVSR.

Set `LOAD_MODEL` to `False` in `config.py`. Set the number of epochs to 150 and the learning rate to 1e-4.
Within `train.py`, remove the discriminator from the training process by commenting it out. Additionally,
the generator should only use MSE loss. Start the training process by running `python3 train.py`. 

Once training is finished, set `LOAD_MODEL` to `True` in `config.py`. The number of epochs and the learning rate
are the same as the previous run. Include the discriminator by uncommenting the previously commented out 
lines of code. The generator should now be using VGG loss, MSE loss, and adverserial loss. Once again run `python3 train.py` 

For the last part of training, change the learning rate to 1e-5 and run `python3 train.py`. Training should now be
complete.

# How to run `test.py`

Make sure the RealVSR testing dataset is located at `datasets/RealVSR/test/`. Then run `python3 test.py`.
