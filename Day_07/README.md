# Machine Learning Challenge: Day 7
Welcome to the seventh day of our 30 days Machine learning Challenge.


At the end of this step, you will understand the concepts of underfitting and overfitting, and you will be able to apply these ideas to make your models more accurate.


### Experimenting With Different Models


Now that you have a reliable way to measure model accuracy, you can experiment with alternative models and see which gives the best predictions. But what alternatives do you have for models?


You can see in scikit-learn's documentation that the decision tree model has many options (more than you'll want or need for a long time). The most important options determine the tree's depth. Recall from the third lesson in this course that a tree's depth is a measure of how many splits it makes before coming to a prediction. This is a relatively shallow tree:


![image](https://user-images.githubusercontent.com/93423367/212739286-79eb15ff-af3d-40ad-b06d-385a079694c7.png)


In practice, it's not uncommon for a tree to have 10 splits between the top level (all houses) and a leaf. The dataset gets sliced up into leaves with fewer houses as the tree gets deeper. If a tree only had 1 split, it divides the data into 2 groups. If each group is split again, we would get 4 groups of houses. Splitting each of those again would create 8 groups. If we keep doubling the number of groups by adding more splits at each level, we'll have 210210 groups of houses by the time we get to the 10th level. That's 1024 leaves.


When we divide the houses amongst many leaves, we also have fewer houses in each leaf. Leaves with very few houses will make predictions that are quite close to those homes' actual values, but they may make very unreliable predictions for new data (because each prediction is based on only a few houses).
This is a phenomenon called overfitting, where a model matches the training data almost perfectly, but does poorly in validation and other new data. On the flip side, if we make our tree very shallow, it doesn't divide up the houses into very distinct groups.


At an extreme, if a tree divides houses into only 2 or 4, each group still has a wide variety of houses. Resulting predictions may be far off for most houses, even in the training data (and it will be bad in validation too for the same reason). When a model fails to capture important distinctions and patterns in the data, so it performs poorly even in training data, that is called underfitting.


Since we care about accuracy on new data, which we estimate from our validation data, we want to find the sweet spot between underfitting and overfitting. Visually, we want the low point of the (red) validation curve in the figure below.


![image](https://user-images.githubusercontent.com/93423367/212739376-60835f04-5252-4c35-babd-d00b598419f9.png)


Example
There are a few alternatives for controlling the tree depth, and many allow for some routes through the tree to have greater depth than other routes. But the max_leaf_nodes argument provides a very sensible way to control overfitting vs underfitting. The more leaves we allow the model to make, the more we move from the underfitting area in the above graph to the overfitting area.
We can use a utility function to help compare MAE scores from different values for max_leaf_nodes:

```
from sklearn.metrics import mean_absolute_error

from sklearn.tree import DecisionTreeRegressor

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):

	model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)

	model.fit(train_X, train_y)

	preds_val = model.predict(val_X)

	mae = mean_absolute_error(val_y, preds_val)

	return(mae)
```


The data is loaded into train_X, val_X, train_y and val_y using the code you've already seen (and which you've already written).

```

# Data Loading Code Runs At This Point

import pandas as pd

# Load data (source: https://github.com/njtierney/melb-housing-data/blob/master/data/housing.csv"

melbourne_file_path = 'housing.csv'

melbourne_data = pd.read_csv(melbourne_file_path)

# Filter rows with missing values

filtered_melbourne_data = melbourne_data.dropna(axis=0)

# Choose target and features

y = filtered_melbourne_data.Price

melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea','YearBuilt', 'Lattitude', 'Longtitude']

X = filtered_melbourne_data[melbourne_features]


from sklearn.model_selection import train_test_split

# split data into training and validation data, for both features and target

train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)


```


We can use a for-loop to compare the accuracy of models built with different values for max_leaf_nodes.

In [3]:

```
# compare MAE with differing values of max_leaf_nodes

for max_leaf_nodes in [5, 50, 500, 5000]:

	my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)

	print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))

Max leaf nodes: 5  		 Mean Absolute Error:  347380 

Max leaf nodes: 50  		 Mean Absolute Error:  258171 

Max leaf nodes: 500  		 Mean Absolute Error:  243495 

Max leaf nodes: 5000            Mean Absolute Error:  254983 


```



Of the options listed, 500 is the optimal number of leaves.


Here's the takeaway: Models can suffer from either:


- Overfitting: capturing spurious patterns that won't recur in the future, leading to less accurate predictions, or

- Underfitting: failing to capture relevant patterns leads to less accurate predictions.

We use validation data, which isn't used in model training, to measure a candidate model's accuracy. This lets us try many candidate models and keep the best one.


Decision trees leave you with a difficult decision. A deep tree with lots of leaves will overfit because each prediction is coming from historical data from only the few houses at its leaf. But a shallow tree with few leaves will perform poorly because it fails to capture as many distinctions in the raw data.


Even today's most sophisticated modeling techniques face this tension between underfitting and overfitting. But, many models have clever ideas that can lead to better performance. We'll look at the random forest as an example.


The random forest uses many trees and makes a prediction by averaging the predictions of each component tree. It generally has much better predictive accuracy than a single decision tree and it works well with default parameters. If you keep modeling, you can learn more models with even better performance, but many of those are sensitive to getting the right parameters.
Example
You've already seen the code to load the data a few times. At the end of data-loading, we have the following variables:


- train_X

- val_X

- train_y

- val_y




In [1]:

```
import pandas as pd

# Load datamelbourne_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'

melbourne_data = pd.read_csv(melbourne_file_path)

# Filter rows with missing valuesmelbourne_data = melbourne_data.dropna(axis=0)

# Choose target and featuresy = melbourne_data.Pricemelbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea','YearBuilt', 'Lattitude', 'Longtitude']X = melbourne_data[melbourne_features]

from sklearn.model_selection import train_test_split

# split data into training and validation data, for both features and target

# The split is based on a random number generator. Supplying a numeric value to

# the random_state argument guarantees we get the same split every time we

# run this script.train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)
```

 #
We build a random forest model similarly to how we built a decision tree in scikit-learn - this time using the RandomForestRegressor class instead of DecisionTreeRegressor.
 
In [2]:
```

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

forest_model = RandomForestRegressor(random_state=1)

forest_model.fit(train_X, train_y)

melb_preds = forest_model.predict(val_X)

print(mean_absolute_error(val_y, melb_preds))

191669.7536453626 

```

Conclusion


There is likely room for further improvement, but this is a big improvement over the best decision tree error of 250,000. There are parameters that allow you to change the performance of the Random Forest much as we changed the maximum depth of the single decision tree. But one of the best features of Random Forest models is that they generally work reasonably even without this tuning.

