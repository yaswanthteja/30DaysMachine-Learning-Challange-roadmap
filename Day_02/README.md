
## Machine Learning Challenge: Day 2

Welcome to the second day of our 30 days Machine learning Challenge.


### Here is the Project One: Classification Walkthrough using Titanic Dataset:


The data has been split into two groups:

- training set (train.csv)

- test set(test.csv)


The training set includes passengers' survival status(also known as the ground truth from the titanic tragedy), which along with other features like gender, class, fare, and pclass is used to create a machine learning model.

The test set should be used to see how well my model performs on unseen data. The test set does not provide passengers with survival status. We are going to use our model to predict passenger survival status.

Here is the link to the walkthrough of the project along with the [code](https://colab.research.google.com/drive/1sfdeyFMHESzjrzVijzOOTh7qcungduvT?usp=sharing)

Let's describe what's the meaning of the features given in both train & test datasets.


Variable Definition Key.
- Survival

  0= No

  1= Yes

- pclass (Ticket class)

  1=1st

  2=2nd

  3=3rd

- sex

- age

- sibsp (# of siblings / spouses aboard the Titanic)

- parch (# of parents / children aboard the Titanic)

- tickets

- fare

- cabin

- embarked Port of Embarkation.

C = Cherbourg,

Q = Queenstown,

S = Southampton

- pclass: A proxy for socio-economic status (SES)
This is important to remember and will come in handy for later analysis.


1st = Upper

2nd = Middle

3rd = Lower


https://youtu.be/G5m4yu0Xtn4

### Supplementary content:s
 

Supervised Learning is a machine learning and artificial intelligence subcategory. It is distinguished by using labeled datasets to train algorithms that accurately classify data or predict outcomes. As input data is fed into a certain model, the weights are adjusted specifically until the model is properly fitted. This occurs as a part of the cross-validation process. 


Supervised learning examples

There are some very practical applications of supervised learning algorithms in real life, including:

·      Text categorization

·      Face Detection

·      Signature recognition

·      Customer discovery

·      Spam detection

·      Weather forecasting

·      Predicting housing prices based on the prevailing market price

·      Stock price predictions, among others

In supervised Learning, a training set is used to teach models how to make the right output. This training dataset has both right and wrong answers, which helps the model learn over time. Using the loss function, the algorithm figures out how accurate it is and changes until the error is as small as possible.


### Challenges of supervised Learning
Although supervised Learning can provide businesses with benefits such as deep data insights and improved automation, building long-term supervised learning models can be difficult. Some of these challenges are as follows:

- Specific levels of expertise may be required for structured, supervised learning models.

- Training supervised learning models takes time. 

- Datasets may contain more human error, causing algorithms to learn incorrectly. 

- Unlike unsupervised learning models, supervised Learning cannot independently cluster or classify data


Supervised Learning can be divided into two types of problems: 

1.     Classification and 

2.     Regression



