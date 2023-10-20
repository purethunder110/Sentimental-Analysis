# Sentimental analysis
 A simple flask and ML program to tell a sentence is positive or negative

 # Setup
 It requires to have python installed in your system.
 
 #### Step 1:
  First clone the repository<br>
```git clone```

#### Step 2:
open cmd/terminal on the folder of the cloned repositories

#### Step 3:
run the following command<br>

```pip install -r requirements.txt```

this command will install all the dependencies that are required for the application to run

# How to run

Sfter following above instruction, just run the ```main.py``` file to run the application.<br>
The application will run on the ```127.0.0.1:8000``` and can now be accessed on the browser.


 # How to train this through your own dataset

 the currenmt model uses "IMDB 2011 top movie reviews" dataset with 25k training and testing dataset divided into positive and negative reviews , in total 50k dataset

 change to the cloned directory and run <br>
 ```python main.py```

put the train and test dataset in data/dataset in the label with clear test and train label, like ```data/dataset/train``` and ```data/dataset/test```.<br><br>
 After processing and cleaning the dataset, run the model.py file to make your new model

# Explaination

The model creates a TF-IDF vectorizer to create a representation for the data to be processed by a simple SVM classifier for a classification.<br>

We then dump our TF-IDF model to the vector.bin and our classifier model to classifier.dump using joblib.

the reason to dump our TDF-IDF vectorizer is so that the new prompts that are getting sent to the model have to be vectorized to the same function so that it can give a similar result.<br>

These dump files are then loaded on a wrapper class in the ```classifier_model.py``` file and no can be use to call from anywhere in the program
