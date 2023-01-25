


# Machine Learning Challenge: Day 8


## Feature Engineering In Machine Learning

![image](https://user-images.githubusercontent.com/93423367/214635260-6b86899a-ea75-4361-805d-faf339a663f6.png)



### Feature selection

Feature selection is the process of identifying which features are most important in your data. If you can find a way to remove unimportant features, then it's possible that you'll be able to improve the performance of your machine-learning model. Feature selection can be done via many different techniques, and there are many different ways to go about it. In this post, we will cover some of these techniques and explain why they work so well for feature selection:

 

### Collinear Columns

Collinear columns are two columns that are highly correlated. If you have several input variables, and one of the inputs is strongly related to another input, it can be hard to tell which is causing the effect. For example, if you have two features (A and B) with a high correlation between them, then it’s hard to tell whether A causes B or vice versa.

In order for us humans to use our eyesight effectively in everyday life—like when we look at pictures—we need all sorts of different kinds of information: color vision helps us recognize shapes, contrast vision helps us spot small details like the text on a page, etc.; however, this means that there may be some overlap between how our eyes work together so that what one person sees as blue might actually just look more yellow than pink for someone else!




### Lasso Regression

Lasso Regression is a type of regression used to find the coefficients of a linear model that best fits a set of data. It uses a regularization parameter (lambda), AIC, or BIC as criteria for performing machine learning on each sample point to select the optimal model parameter that minimizes error while not overfitting.

The lasso method makes use of regularization techniques to control the strength of penalty on coefficients, which allows it to capture nonlinear relationships between variables in your dataset and make predictions about them.




### Recursive Feature Elimination

Recursive feature elimination (RFE), also known as recursive feature selection and recursive partitioning, is a greedy algorithm that iteratively finds the optimal subset of features to be used in an overall model. It's a form of forward selection and backward elimination: you first select a subset of training data and then remove some features from your model until you're left with only those that improve its performance on testing data.

The process works like this: first, you choose all possible subsets based on some criterion; then, for each possible subset, remove one or more features from your model depending on how well they have been selected already; repeat until either no other good subsets remain or there aren't any leftovers!

 

### Mutual Information

- Mutual information is a measure of the dependence between two random variables. It can be computed in many ways, but we'll use one simple method:

  - Mutual information I(x) = -Pr(x|y)Pr(y|x)
   - Mutual Information measures how much an average value from one variable depends on an average value from another variable. In other words, it tells us how much more likely it is for both variables to change together than for them to change separately. For example, suppose your company wants to know how important sales will be this year compared with last year's performance. In that case, you'd want to look at how well each product performed last year compared with its own historical performance (i.e., whether it did better or worse). By comparing these two numbers against each other—and then subtracting them—you'll get an accurate estimate of whether your products are performing well enough right now so that they'll continue doing well in future years as well!




### Principal Component Analysis

Principal Component Analysis (PCA) is a dimension reduction technique that reduces the number of features in a dataset without losing information. PCA finds the directions (principal components) that maximize the variance of your data set.

Principal Components Analysis can determine the most important features for predicting an outcome, like how many people will buy your product or service.

The idea behind PCA is that you can reduce the dimensionality of your data by projecting it onto a new set of axes that are linear combinations of the original features. The new set of axes will be uncorrelated from each other, and each axis will be orthogonal to all other axes. This means that each axis represents a different aspect of your data set, allowing you to visualize these aspects separately.




### Feature Importance

Feature importance is a term used to describe the relative importance of each feature for predicting the outcome. For example, if you have 10 features and your model is using just 1 or 2 of them, it means that those features are not very important in predicting your target variable. To avoid this problem, you could use a smaller number of more relevant variables instead.

You can also think about it as an indicator for which columns should be included in your model when using machine learning techniques such as linear regression or neural networks (which we'll talk about later). If there are several columns with similar values but different sizes, you might leave out some columns because they add little information compared to others that provide more value in making predictions on new data sets (or even existing ones).



### Conclusion

I explained how to use feature selection in machine learning in this article. Feature selection is a powerful tool that can help you improve your model's performance by removing unimportant features. By examining the data with different tools and finding out which features are important for your problem, you can end up with a better model that has more relevant information about what is important in reality. Before using these methods for real-world applications, you should try them out on your own problems!

































