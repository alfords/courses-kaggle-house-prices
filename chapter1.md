---
title       : Chapter 1
description : Stuff and stuff
attachments :

--- type:NormalExercise lang:python xp:100 skills:1 key:ca2686ceeb
## How it works

Welcome to our Kaggle House Prices: Advanced Regression tutorial. In this tutorial, you will explore how to tackle the Kaggle House Prices: Advanced Regression Techniques competition using Python and Machine Learning. In case you're new to Python, it's recommended that you first take our free <a target="_blank" href="https://www.datacamp.com/courses/intro-to-python-for-data-science"> Introduction to Python for Data Science<a/> Tutorial. Furthermore, while not required, familiarity with machine learning techniques is a plus so you can get the maximum out of this tutorial.

In the editor on the right, you should type Python code to solve the exercises. When you hit the 'Submit Answer' button, every line of code is interpreted and executed by Python and you get a message whether or not your code was correct. The output of your Python code is shown in the console in the lower right corner. Python makes use of the `#` sign to add comments; these lines are not run as Python code, so they will not influence your result.

You can also execute Python commands straight in the console. This is a good way to experiment with Python code, as your submission is not checked for correctness.

*** =instructions
- In the editor to the right, you see some Python code and annotations. This is what a typical exercise will look like.
- To complete the exercise and see how the interactive environment works add the code to compute y and hit the `Submit Answer` button. Don't forget to print the result.

*** =hint
- Just add a line of Python code that calculates the product of 6 and 9, just like the example in the sample code!

*** =pre_exercise_code
```{python}
```

*** =sample_code
```{python}
#Compute x = 4 * 3 and print the result
x = 4 * 3
print(x)

#Compute y = 6 * 9 and print the result
y = 6*9; print(y)

```

*** =solution
```{python}
#Compute x = 4 * 3 and print the result
x = 4 * 3
print(x)

#Compute y = 6 * 9 and print the result
y = 6*9; print(y)
```

*** =sct
```{python}
msg = "Don't forget to assign the correct value to y"
test_object("y", 
            undefined_msg = msg, 
            incorrect_msg = msg)

msg = "Print out the resulting object, `y`!"
test_function("print",2, 
              not_called_msg = msg,
              incorrect_msg = msg,
              args=None)

success_msg("Awesome! See how the console shows the result of the Python code you submitted? Now that you're familiar with the interface, let's get down to business!")
```

--- type:NormalExercise lang:python xp:100 skills:2 key:672930f088
## Get the data with Pandas 
For many the dream of owning a home doesn't include searching for the perfect basement ceiling height or the proximity to an east-west railroad. However, the 79 explanatory variables describing (almost) every aspect of residential homes used in the Kaggle House Price Competition show that there is much more that influences price negotiations than the number of bedrooms or a white-picket fence.

In this course, you will learn how to apply machine learning techniques to predict the final price of each home using Python.

"The potential for creative feature engineering provides a rich opportunity for fun and learning. This dataset lends itself to advanced regression techniques like random forests and gradient boosting with the popular XGBoost library. We encourage Kagglers to create benchmark code and tutorials on Kernels for community learning. Top kernels will be awarded swag prizes at the competition close." 

*** =instructions
- First, import the Pandas library as pd.
- Load the test data similarly to how the train data is loaded.
- Inspect the first couple rows of the loaded dataframes using the .head() method with the code provided.

*** =hint
- You can load in the training set with ```train = pd.read_csv(train_url)```
- To print a variable to the console, use the print function on a new line.

*** =pre_exercise_code
```{python}

```

*** =sample_code
```{python}
# Import the Pandas library
import pandas as pd

# Load the train and test datasets to create two DataFrames
train_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
train = pd.read_csv(train_url)

test_url = "https://s3.amazonaws.com/assets.datacamp.com/production/course_2470/datasets/test.csv"
test = pd.read_csv(test_url)

#Print the `head` of the train and test dataframes
print(train.head())
print(test.head())
```

*** =solution
```{python}
# Import the Pandas library
import pandas as pd

# Load the train and test datasets to create two DataFrames
train_url = "https://s3.amazonaws.com/assets.datacamp.com/production/course_2470/datasets/train.csv"
train = pd.read_csv(train_url)

test_url = "https://s3.amazonaws.com/assets.datacamp.com/production/course_2470/datasets/test.csv"
test = pd.read_csv(test_url)

#Print the `head` of the train and test dataframes
print(train.head())
print(test.head())
```

*** =sct
```{python}
msg = "Have you correctly imported the `pandas` package? Use the alias `pd`."
test_import("pandas",  not_imported_msg = msg,  incorrect_as_msg = msg)

msg = "Do not touch the code that specifies the URLs of the training and test set csvs."
test_object("train_url", undefined_msg = msg, incorrect_msg = msg)
test_object("test_url", undefined_msg = msg, incorrect_msg = msg)

msg = "Make sure you are using the `read_csv()` function correctly"
test_function("pandas.read_csv", 1,
              args=None,
              not_called_msg = msg,
              incorrect_msg = msg,)
test_function("pandas.read_csv", 2,
              args=None,
              not_called_msg = msg,
              incorrect_msg = msg)

#msg = "Don't forget to print the first few rows of `train` with the `.head()` method"
#test_function("print", 1, not_called_msg = msg, incorrect_msg = msg)

#msg = "Don't forget to print the first few rows of `test` with the `.head()` method"
#test_function("print", 2, not_called_msg = msg, incorrect_msg = msg)

success_msg("Well done! Now that your data is loaded in, let's see if you can understand it.")
```

--- type:MultipleChoiceExercise lang:python xp:50 skills:2 key:5e47ef16d2
## <<<New Exercise>>> 

Before starting with the actual analysis, it's important to understand the structure of your data. Both `test` and `train` are DataFrame objects, the way pandas represent datasets. You can easily explore a DataFrame using the `.describe()` method. `.describe()` summarizes the columns/features of the DataFrame, including the count of observations, mean, max and so on. Another useful trick is to look at the dimensions of the DataFrame. This is done by requesting the `.shape` attribute of your DataFrame object. (ex. `your_data.shape`)

The training and test set are already available in the workspace, as `train` and `test`. Apply `.describe()` method and print the `.shape` attribute of the training set. Which of the following statements is correct?

*** =instructions
- The training set has 1460 observations and 81 variables, count for LotFrontage is 1233.
- The training set has 1459 observations and 80 variables, count for LotFrontage is 1459.
- The testing set has 1459 observations and 81 variables, count for LotFrontage is 1234.
- The testing set has 1459 observations and 80 variables, count for LotFrontage is 1232.

*** =hint
To see the description of the `test` variable try `test.describe()`.

*** =pre_exercise_code
```{python}
import pandas as pd
train = pd.read_csv("https://s3.amazonaws.com/assets.datacamp.com/production/course_2470/datasets/train.csv")
test = pd.read_csv("https://s3.amazonaws.com/assets.datacamp.com/production/course_2470/datasets/test.csv")
```

*** =sct

```{python}

msg1 = "Incorrect. Maybe have a look at the hint."
msg2 = "Wrong, try again. Maybe have a look at the hint."
msg3 = "Not so good... Maybe have a look at the hint."
msg4 = "Great job!"
test_mc(correct = 4, msgs = [msg1, msg2, msg3, msg4])

success_msg("Well done! Now move on and explore some of the features in more detail.")

```

