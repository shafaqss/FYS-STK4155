### Information about the data sets:
I have used two data sets. The first one is the wisconsin breast cancer data. This can easily be loaded form sklearn in python. Here is how I have loaded it in python

```
from sklearn.datasets import load_breast_cancer
bc_data = load_breast_cancer()

# design matrix, with shape (569, 30)
X = pd.DataFrame(bc_data.data, columns=bc_data.feature_names)
#targets/labels with shape (569,)
y = pd.DataFrame(bc_data.target)
y = y.iloc[:,0]
```
The second data set is the south african heart disease data set. This is contained in the file ```SAheart.txt```. I got this from the [website](https://web.stanford.edu/~hastie/ElemStatLearn/). You have to click on South African Heart Disease. For information about the data go [here](https://web.stanford.edu/~hastie/ElemStatLearn/datasets/SAheart.info.txt), for the data set itself, here is the [link](https://web.stanford.edu/~hastie/ElemStatLearn/datasets/SAheart.data)

### Information about the scripts:
All the scripts with ```breast_cancer``` analyse the breast cancer data, all the scripts with ```heart_disease``` analyse the heart disease data. Here is what each script contains;
* breast_cancer1.py - Loading the breast cancer data, scaling and splitting into test and train
* breast_cancer2.py - Performing logistic regression 
* breast_cancer3.py - Performing logistic regression but with my own code, this is the same logitic regression code that I used in project 2
* breast_cancer4.py - Using the decision tree on the data
* breast_cancer5.py - Using the ensemble methods on the data
* heart_disease1.py - Reading the data, scaling and splitting into test and train
* heart_disease2.py - Performing logistic regression
* heart_disease3.py - Using the decision tree on the data
* heart_disease4.py - Using the ensemble methods on the data
* example_logistic.py, logistic.py, neededfunc.py -  These are the same from my project 2, they just implement logistic regression with my own code.

### Information about the folder Ipython_notebook:
I have mainly used these python scripts in my analysis. However if you want to see an easy visualiztion of my results then you can see the ipython notebooks in the folder ```Ipython_notebook```. Each data set is shown in the two ipython notebooks, ```Breast cancer data analysis.ipynb``` and  ```South Africa heart disease analysis.ipynb```. The folder ```Trash Code``` can just be ignored.
