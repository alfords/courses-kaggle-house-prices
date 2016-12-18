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
import pandas as pd
test = pd.read_csv("https://s3.amazonaws.com/assets.datacamp.com/production/course_2470/datasets/test.csv")

import numpy as np
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
# SCT written with pythonwhat: https://github.com/datacamp/pythonwhat/wiki


success_msg("Great work!")
```
