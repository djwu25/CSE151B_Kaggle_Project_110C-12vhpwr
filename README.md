# CSE151B_Kaggle_Project_110C-12vhpwr

How to run our PyTorch models:  
Our best model for the public score is the `718_4312.pth` file located [here](https://drive.google.com/file/d/1ybiLvI3oBh1as6sM6JoiTelYp3hc7s_P/view?usp=sharing)  
To run the model, edit the `verify.py` file under `./joshua/718-4312-width-sigmoid` to load the pth file and run it with the `tripid.txt`located [here](https://drive.google.com/file/d/1OgAmLWwjw_Q5e1AuerjbioiVftF5G5wO/view?usp=sharing) and test data located [here](https://drive.google.com/file/d/1u0-Zm_NIHpYBej9dUCkZ7oQtuyPAsnrn/view?usp=sharing)  
Alternatively, just rename the `.pth` file to `save.pth` and run `verify.py` in the same directory alongside the `tripid.txt` and test data file.

The same process applies for running any of our other PyTorch models. All the resources to run the models can be found under the following link: [https://drive.google.com/drive/folders/1bOJOv8QjK5yz4oFlXlMGRRT-qFZ8VovM?usp=sharing](https://drive.google.com/drive/folders/1bOJOv8QjK5yz4oFlXlMGRRT-qFZ8VovM?usp=sharing)

Each pytorch directory contains a `competition.py` which was the python script used for training and a `verify.py` which was used for generating the test data csv. The directories are prefixed with an approximation of their public score.

To train a model with our data, have the corresponding data file located in the above directory and run `competition.py`
