# Machine Learning Challenge: Day 4
Welcome to the fourth day of our 30 days Machine learning Challenge.



Pandas is a powerful and flexible open-source data manipulation and analysis library for the Python programming language. It provides data structures such as DataFrame and Series, which are designed to work with large and complex datasets. By using pandas, you can easily get familiar with your data by performing tasks such as:

Loading data from various file formats (e.g. CSV, Excel, JSON)
Exploring and summarizing your data (e.g. head, tail, describe)
Cleaning and transforming your data (e.g. filling in missing values, converting data types)
Filtering and selecting specific rows and columns
Grouping and aggregating your data
Sorting and ordering your data
Merging and joining multiple datasets
One of the most important features of pandas its ability to handle missing data. It provides various methods to handle missing data, such as filling missing values with a specific value or using interpolation techniques.

Overall, pandas is a great tool for data exploration and manipulation, and it's widely used by data scientists and analysts in various industries.

Find the Dataset for this notebook from here
Here's what we're going to do today:

- Take a first look at the data
- See how many missing data points we have
- Figure out why the data is missing
- Drop missing values
- Filling in missing values
Let's get started!


### Using Pandas to Get Familiar With Your Data

The first step in any machine learning project is to familiarize yourself with the data. You'll use the Pandas library for this. Pandas is the primary tool data scientists use for exploring and manipulating data. Most people abbreviate pandas in their code as pd. We do this with the command

```
import pandas as pd
```

The most important part of the Pandas library is the DataFrame. A DataFrame holds the type of data you might think of as a table. This is similar to a sheet in Excel, or a table in a SQL database.


Pandas has powerful methods for most things you'll want to do with this type of data.


As an example, we'll look at data about home prices in Melbourne, Australia. In the hands-on exercises, you will apply the same processes to a new dataset, which has home prices in Iowa.
The example (Melbourne) data can be found at ''https://github.com/njtierney/melb-housing-data/blob/master/data/housing.csv''
We load and explore the data with the following commands:

```

# save filepath to variable for easier access. change file path accordingly
melbourne_file_path = 'housing.csv'
# read the data and store data in DataFrame titled melbourne_data
melbourne_data = pd.read_csv(melbourne_file_path)
# print a summary of the data in Melbourne data
melbourne_data.describe()

```


![image](https://user-images.githubusercontent.com/93423367/212376570-609d488a-60d7-453d-925a-1c837cddc4d7.png)

### Interpreting Data Description

The results show 8 numbers for each column in your original dataset. The first number, the count, shows how many rows have non-missing values.


Missing values arise for many reasons. For example, the size of the 2nd bedroom wouldn't be collected when surveying a 1-bedroom house. We'll come back to the topic of missing data.


The second value is the mean, which is the average. Under that, std is the standard deviation, which measures how numerically spread out the values are.


To interpret the min, 25%, 50%, 75%, and max values, imagine sorting each column from the lowest to highest. The first (smallest) value is the min. If you go a quarter way through the list, you'll find a number that is bigger than 25% of the values and smaller than 75% of the values. That is the 25% value (pronounced "25th percentile"). The 50th and 75th percentiles are defined analogously, and the max is the largest number.



