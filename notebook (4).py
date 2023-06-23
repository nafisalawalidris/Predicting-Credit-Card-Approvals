#!/usr/bin/env python
# coding: utf-8

# ## 1. Credit card applications
# <p>Commercial banks receive <em>a lot</em> of applications for credit cards. Many of them get rejected for many reasons, like high loan balances, low income levels, or too many inquiries on an individual's credit report, for example. Manually analyzing these applications is mundane, error-prone, and time-consuming (and time is money!). Luckily, this task can be automated with the power of machine learning and pretty much every commercial bank does so nowadays. In this notebook, we will build an automatic credit card approval predictor using machine learning techniques, just like the real banks do.</p>
# <p><img src="https://assets.datacamp.com/production/project_558/img/credit_card.jpg" alt="Credit card being held in hand"></p>
# <p>We'll use the <a href="http://archive.ics.uci.edu/ml/datasets/credit+approval">Credit Card Approval dataset</a> from the UCI Machine Learning Repository. The structure of this notebook is as follows:</p>
# <ul>
# <li>First, we will start off by loading and viewing the dataset.</li>
# <li>We will see that the dataset has a mixture of both numerical and non-numerical features, that it contains values from different ranges, plus that it contains a number of missing entries.</li>
# <li>We will have to preprocess the dataset to ensure the machine learning model we choose can make good predictions.</li>
# <li>After our data is in good shape, we will do some exploratory data analysis to build our intuitions.</li>
# <li>Finally, we will build a machine learning model that can predict if an individual's application for a credit card will be accepted.</li>
# </ul>
# <p>First, loading and viewing the dataset. We find that since this data is confidential, the contributor of the dataset has anonymized the feature names.</p>

# In[180]:


# Import pandas
import pandas as pd

# Load dataset
cc_apps = pd.read_csv("datasets/cc_approvals.data", header=None)

# Inspect data
print(cc_apps.head())


# In[181]:


get_ipython().run_cell_magic('nose', '', 'import pandas as pd\n\ndef test_cc_apps_exists():\n    assert "cc_apps" in globals(), \\\n        "The variable cc_apps should be defined."\n        \ndef test_cc_apps_correctly_loaded():\n    correct_cc_apps = pd.read_csv("datasets/cc_approvals.data", header=None)\n    try:\n        pd.testing.assert_frame_equal(cc_apps, correct_cc_apps)\n    except AssertionError:\n        assert False, "The variable cc_apps should contain the data as present in datasets/cc_approvals.data."')


# ## 2. Inspecting the applications
# <p>The output may appear a bit confusing at its first sight, but let's try to figure out the most important features of a credit card application. The features of this dataset have been anonymized to protect the privacy, but <a href="http://rstudio-pubs-static.s3.amazonaws.com/73039_9946de135c0a49daa7a0a9eda4a67a72.html">this blog</a> gives us a pretty good overview of the probable features. The probable features in a typical credit card application are <code>Gender</code>, <code>Age</code>, <code>Debt</code>, <code>Married</code>, <code>BankCustomer</code>, <code>EducationLevel</code>, <code>Ethnicity</code>, <code>YearsEmployed</code>, <code>PriorDefault</code>, <code>Employed</code>, <code>CreditScore</code>, <code>DriversLicense</code>, <code>Citizen</code>, <code>ZipCode</code>, <code>Income</code> and finally the <code>ApprovalStatus</code>. This gives us a pretty good starting point, and we can map these features with respect to the columns in the output.   </p>
# <p>As we can see from our first glance at the data, the dataset has a mixture of numerical and non-numerical features. This can be fixed with some preprocessing, but before we do that, let's learn about the dataset a bit more to see if there are other dataset issues that need to be fixed.</p>

# In[182]:


# Print summary statistics
cc_apps_description = cc_apps.describe()
print(cc_apps_description)

print('\n')

# Print DataFrame information
cc_apps_info = cc_apps.info()
print(cc_apps_info)

print('\n')

# Inspect missing values in the dataset
print(cc_apps.tail(17))


# In[183]:


get_ipython().run_cell_magic('nose', '', '\ndef test_cc_apps_description_exists():\n    assert "cc_apps_description" in globals(), \\\n        "The variable cc_apps_description should be defined."\n\ndef test_cc_apps_description_correctly_done():\n    correct_cc_apps_description = cc_apps.describe()\n    assert str(correct_cc_apps_description) == str(cc_apps_description), \\\n        "cc_apps_description should contain the output of cc_apps.describe()."\n    \ndef test_cc_apps_info_exists():\n    assert "cc_apps_info" in globals(), \\\n        "The variable cc_apps_info should be defined."\n\ndef test_cc_apps_info_correctly_done():\n    correct_cc_apps_info = cc_apps.info()\n    assert str(correct_cc_apps_info) == str(cc_apps_info), \\\n        "cc_apps_info should contain the output of cc_apps.info()."')


# ## 3. Splitting the dataset into train and test sets
# <p>Now, we will split our data into train set and test set to prepare our data for two different phases of machine learning modeling: training and testing. Ideally, no information from the test data should be used to preprocess the training data or should be used to direct the training process of a machine learning model. Hence, we first split the data and then preprocess it.</p>
# <p>Also, features like <code>DriversLicense</code> and <code>ZipCode</code> are not as important as the other features in the dataset for predicting credit card approvals. To get a better sense, we can measure their <a href="https://realpython.com/numpy-scipy-pandas-correlation-python/">statistical correlation</a> to the labels of the dataset. But this is out of scope for this project. We should drop them to design our machine learning model with the best set of features. In Data Science literature, this is often referred to as <em>feature selection</em>. </p>

# In[184]:


# Import train_test_split
from sklearn.model_selection import train_test_split

# Drop features 11 and 13
cc_apps = cc_apps.drop([11, 13], axis=1)

# Split into train and test sets
cc_apps_train, cc_apps_test = train_test_split(cc_apps, test_size=0.33, random_state=42)


# In[185]:


get_ipython().run_cell_magic('nose', '', '\n\ndef test_columns_dropped_correctly():\n    assert cc_apps.shape == (\n        690,\n        14,\n    ), "The shape of the DataFrame isn\'t correct. Did you drop the two columns?"\n\n\ndef test_data_split_correctly():\n    cc_apps_train_correct, cc_apps_test_correct = train_test_split(\n        cc_apps, test_size=0.33, random_state=42\n    )\n    assert cc_apps_train_correct.equals(cc_apps_train) and cc_apps_test_correct.equals(\n        cc_apps_test\n    ), "It doesn\'t appear that the data splitting was done correctly."')


# ## 4. Handling the missing values (part i)
# <p>Now we've split our data, we can handle some of the issues we identified when inspecting the DataFrame, including:</p>
# <ul>
# <li>Our dataset contains both numeric and non-numeric data (specifically data that are of <code>float64</code>, <code>int64</code> and <code>object</code> types). Specifically, the features 2, 7, 10 and 14 contain numeric values (of types float64, float64, int64 and int64 respectively) and all the other features contain non-numeric values.</li>
# <li>The dataset also contains values from several ranges. Some features have a value range of 0 - 28, some have a range of 2 - 67, and some have a range of 1017 - 100000. Apart from these, we can get useful statistical information (like <code>mean</code>, <code>max</code>, and <code>min</code>) about the features that have numerical values. </li>
# <li>Finally, the dataset has missing values, which we'll take care of in this task. The missing values in the dataset are labeled with '?', which can be seen in the last cell's output of the second task.</li>
# </ul>
# <p>Now, let's temporarily replace these missing value question marks with NaN.</p>

# In[186]:


# Import numpy
import numpy as np

# Replace the '?'s with NaN in the train and test sets
cc_apps_train = cc_apps_train.replace('?', np.nan)
cc_apps_test = cc_apps_test.replace('?', np.nan)


# In[187]:


get_ipython().run_cell_magic('nose', '', '\n# def test_cc_apps_assigned():\n#     assert "cc_apps" in globals(), \\\n#         "After the NaN replacement, it should be assigned to the same variable cc_apps only."\n\n\ndef test_cc_apps_correctly_replaced():\n    cc_apps_fresh = pd.read_csv("datasets/cc_approvals.data", header=None)\n    cc_apps_train_correct, cc_apps_test_correct = train_test_split(\n        cc_apps_fresh, test_size=0.33, random_state=42\n    )\n\n    correct_cc_apps_replacement_correct_train = cc_apps_train_correct.replace(\n        "?", np.NaN\n    )\n    correct_cc_apps_replacement_correct_test = cc_apps_test_correct.replace("?", np.NaN)\n    string_cc_apps_replacement_train = cc_apps_train_correct.replace("?", "NaN")\n    string_cc_apps_replacement_test = cc_apps_test_correct.replace("?", "NaN")\n    #     assert cc_apps.to_string() == correct_cc_apps_replacement.to_string(), \\\n    #         "The code that replaces question marks with NaNs doesn\'t appear to be correct."\n    try:\n        pd.testing.assert_frame_equal(\n            correct_cc_apps_replacement_correct_train, cc_apps_train\n        )\n\n        pd.testing.assert_frame_equal(\n            correct_cc_apps_replacement_correct_test, cc_apps_test\n        )\n    except AssertionError:\n        if string_cc_apps_replacement_train.equals(\n            cc_apps_train\n        ) or string_cc_apps_replacement_test.equals(cc_apps_test):\n            assert (\n                False\n            ), \'It looks like the question marks were replaced by the string "NaN". Missing values should be represented by `np.nan`.\'')


# ## 5. Handling the missing values (part ii)
# <p>We replaced all the question marks with NaNs. This is going to help us in the next missing value treatment that we are going to perform.</p>
# <p>An important question that gets raised here is <em>why are we giving so much importance to missing values</em>? Can't they be just ignored? Ignoring missing values can affect the performance of a machine learning model heavily. While ignoring the missing values our machine learning model may miss out on information about the dataset that may be useful for its training. Then, there are many models which cannot handle missing values implicitly such as Linear Discriminant Analysis (LDA). </p>
# <p>So, to avoid this problem, we are going to impute the missing values with a strategy called mean imputation.</p>

# In[188]:


# Impute the missing values with mean imputation
cc_apps_train.fillna(cc_apps_train.mean(), inplace=True)
cc_apps_test.fillna(cc_apps_train.mean(), inplace=True)

# Count the number of NaNs in the datasets and print the counts to verify
print(cc_apps_train.isnull().sum())
print(cc_apps_test.isnull().sum())


# In[189]:


get_ipython().run_cell_magic('nose', '', '\n\ndef test_cc_apps_correctly_imputed():\n    assert (\n        cc_apps_train.isnull().to_numpy().sum() == 39\n        and cc_apps_test.isnull().to_numpy().sum() == 15\n    ), "There should be 39 and 15 null values in the training and test sets after your code is run, but there aren\'t."')


# ## 6. Handling the missing values (part iii)
# <p>We have successfully taken care of the missing values present in the numeric columns. There are still some missing values to be imputed for columns 0, 1, 3, 4, 5, 6 and 13. All of these columns contain non-numeric data and this is why the mean imputation strategy would not work here. This needs a different treatment. </p>
# <p>We are going to impute these missing values with the most frequent values as present in the respective columns. This is <a href="https://www.datacamp.com/community/tutorials/categorical-data">good practice</a> when it comes to imputing missing values for categorical data in general.</p>

# In[190]:


# Iterate over each column of cc_apps_train
for col in cc_apps_train.columns:
    # Check if the column is of object type
    if cc_apps_train[col].dtypes == 'object':
        # Impute with the most frequent value
        cc_apps_train[col] = cc_apps_train[col].fillna(cc_apps_train[col].value_counts().index[0])
        cc_apps_test[col] = cc_apps_test[col].fillna(cc_apps_train[col].value_counts().index[0])

# Count the number of NaNs in the dataset and print the counts to verify
print(cc_apps_train.isnull().sum())
print(cc_apps_test.isnull().sum())


# In[191]:


get_ipython().run_cell_magic('nose', '', '\n\ndef test_cc_apps_correctly_imputed():\n    assert (\n        cc_apps_train.isnull().to_numpy().sum() == 0\n        and cc_apps_test.isnull().to_numpy().sum() == 0\n    ), "There should be 0 null values after your code is run, but there isn\'t."')


# ## 7. Preprocessing the data (part i)
# <p>The missing values are now successfully handled.</p>
# <p>There is still some minor but essential data preprocessing needed before we proceed towards building our machine learning model. We are going to divide these remaining preprocessing steps into two main tasks:</p>
# <ol>
# <li>Convert the non-numeric data into numeric.</li>
# <li>Scale the feature values to a uniform range.</li>
# </ol>
# <p>First, we will be converting all the non-numeric values into numeric ones. We do this because not only it results in a faster computation but also many machine learning models (like XGBoost) (and especially the ones developed using scikit-learn) require the data to be in a strictly numeric format. We will do this by using the <code>get_dummies()</code> method from pandas.</p>

# In[192]:


# Convert the categorical features in the train and test sets independently
cc_apps_train = pd.get_dummies(cc_apps_train)
cc_apps_test = pd.get_dummies(cc_apps_test)

# Reindex the columns of the test set aligning with the train set
cc_apps_test = cc_apps_test.reindex(columns=cc_apps_train.columns, fill_value=0)


# In[193]:


get_ipython().run_cell_magic('nose', '', '\n\ndef test_label_encoding_done_correctly():\n    for col in cc_apps_train.columns:\n        if (\n            np.issubdtype(cc_apps_train[col].dtype, np.number) != True\n            and np.issubdtype(cc_apps_test[col].dtype, np.number) != True\n        ):\n            assert "It doesn\'t appear that all of the non-numeric columns were converted to numeric using fit_transform and transform."')


# ## 8. Preprocessing the data (part ii)
# <p>Now, we are only left with one final preprocessing step of scaling before we can fit a machine learning model to the data. </p>
# <p>Now, let's try to understand what these scaled values mean in the real world. Let's use <code>CreditScore</code> as an example. The credit score of a person is their creditworthiness based on their credit history. The higher this number, the more financially trustworthy a person is considered to be. So, a <code>CreditScore</code> of 1 is the highest since we're rescaling all the values to the range of 0-1.</p>

# In[194]:


# Import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler

# Segregate features and labels into separate variables
X_train, y_train = cc_apps_train.iloc[:, :-1].values, cc_apps_train.iloc[:, -1].values
X_test, y_test = cc_apps_test.iloc[:, :-1].values, cc_apps_test.iloc[:, -1].values

# Instantiate MinMaxScaler and use it to rescale X_train and X_test
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX_train = scaler.fit_transform(X_train)
rescaledX_test = scaler.transform(X_test)


# In[195]:


get_ipython().run_cell_magic('nose', '', '\n\ndef test_training_range_set_correctly():\n    min_value_in_rescaledX_train = np.amin(rescaledX_train)\n    max_value_in_rescaledX_train = np.amax(rescaledX_train)\n    assert (\n        min_value_in_rescaledX_train == 0.0 and max_value_in_rescaledX_train == 1.0\n    ), "Did you correctly fit and transform the `X_train` data?"\n\n\ndef test_xtest_created():\n    assert (\n        "rescaledX_test" in globals()\n    ), "Did you correctly use the fitted `scaler` to transform the `X_test` data?"')


# ## 9. Fitting a logistic regression model to the train set
# <p>Essentially, predicting if a credit card application will be approved or not is a <a href="https://en.wikipedia.org/wiki/Statistical_classification">classification</a> task. According to UCI, our dataset contains more instances that correspond to "Denied" status than instances corresponding to "Approved" status. Specifically, out of 690 instances, there are 383 (55.5%) applications that got denied and 307 (44.5%) applications that got approved. </p>
# <p>This gives us a benchmark. A good machine learning model should be able to accurately predict the status of the applications with respect to these statistics.</p>
# <p>Which model should we pick? A question to ask is: <em>are the features that affect the credit card approval decision process correlated with each other?</em> Although we can measure correlation, that is outside the scope of this notebook, so we'll rely on our intuition that they indeed are correlated for now. Because of this correlation, we'll take advantage of the fact that generalized linear models perform well in these cases. Let's start our machine learning modeling with a Logistic Regression model (a generalized linear model).</p>

# In[196]:


# Import LogisticRegression
from sklearn.linear_model import LogisticRegression

# Instantiate a LogisticRegression classifier with default parameter values
logreg = LogisticRegression()

# Fit logreg to the train set
logreg.fit(rescaledX_train, y_train)


# In[197]:


get_ipython().run_cell_magic('nose', '', '\n\ndef test_logreg_defined():\n    assert (\n        "logreg" in globals()\n    ), "Did you instantiate LogisticRegression in the logreg variable?"\n\n\ndef test_logreg_defined_correctly():\n    logreg_correct = LogisticRegression()\n    assert str(logreg_correct) == str(\n        logreg\n    ), "The logreg variable should be defined with LogisticRegression() only."')


# ## 10. Making predictions and evaluating performance
# <p>But how well does our model perform? </p>
# <p>We will now evaluate our model on the test set with respect to <a href="https://developers.google.com/machine-learning/crash-course/classification/accuracy">classification accuracy</a>. But we will also take a look the model's <a href="http://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/">confusion matrix</a>. In the case of predicting credit card applications, it is important to see if our machine learning model is equally capable of predicting approved and denied status, in line with the frequency of these labels in our original dataset. If our model is not performing well in this aspect, then it might end up approving the application that should have been approved. The confusion matrix helps us to view our model's performance from these aspects.  </p>

# In[198]:


# Import confusion_matrix
from sklearn.metrics import confusion_matrix

# Use logreg to predict instances from the test set and store it
y_pred = logreg.predict(rescaledX_test)

# Get the accuracy score of logreg model and print it
print("Accuracy of logistic regression classifier: ", logreg.score(rescaledX_test, y_test))

# Print the confusion matrix of the logreg model
print(confusion_matrix(y_test, y_pred))


# In[199]:


get_ipython().run_cell_magic('nose', '', '\ndef test_ypred_defined():\n    assert "y_pred" in globals(),\\\n        "The variable y_pred should be defined."\n\ndef test_ypred_defined_correctly():\n    correct_y_pred = logreg.predict(rescaledX_test)\n    assert str(correct_y_pred) == str(y_pred),\\\n        "The y_pred variable should contain the predictions as made by LogisticRegression on rescaledX_test."')


# ## 11. Grid searching and making the model perform better
# <p>Our model was pretty good! In fact it was able to yield an accuracy score of 100%.</p>
# <p>For the confusion matrix, the first element of the of the first row of the confusion matrix denotes the true negatives meaning the number of negative instances (denied applications) predicted by the model correctly. And the last element of the second row of the confusion matrix denotes the true positives meaning the number of positive instances (approved applications) predicted by the model correctly.</p>
# <p>But if we hadn't got a perfect score what's to be done?. We can perform a <a href="https://machinelearningmastery.com/how-to-tune-algorithm-parameters-with-scikit-learn/">grid search</a> of the model parameters to improve the model's ability to predict credit card approvals.</p>
# <p><a href="http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html">scikit-learn's implementation of logistic regression</a> consists of different hyperparameters but we will grid search over the following two:</p>
# <ul>
# <li>tol</li>
# <li>max_iter</li>
# </ul>

# In[200]:


# Import GridSearchCV
from sklearn.model_selection import GridSearchCV

# Define the grid of values for tol and max_iter
tol = [0.01, 0.001, 0.0001]
max_iter = [100, 150, 200]

# Create a dictionary where tol and max_iter are keys and the lists of their values are corresponding values
param_grid = dict(tol=tol, max_iter=max_iter)


# In[201]:


get_ipython().run_cell_magic('nose', '', '\n\ndef test_tol_defined():\n    assert "tol" in globals(), "The variable tol should be defined."\n\n\ndef test_max_iter_defined():\n    assert "max_iter" in globals(), "The variable max_iter should be defined."\n\n\ndef test_tol_defined_correctly():\n    correct_tol = [0.01, 0.001, 0.0001]\n    assert (\n        correct_tol == tol\n    ), "It looks like the tol variable is not defined with the list of correct values."\n\n\ndef test_max_iter_defined_correctly():\n    correct_max_iter = [100, 150, 200]\n    assert (\n        correct_max_iter == max_iter\n    ), "It looks like the max_iter variable is not defined with a list of correct values."\n\n\ndef test_param_grid_defined():\n    assert "param_grid" in globals(), "The variable param_grid should be defined."\n\n\ndef test_param_grid_defined_correctly():\n    correct_param_grid = dict(tol=tol, max_iter=max_iter)\n    assert str(correct_param_grid) == str(\n        param_grid\n    ), "It looks like the param_grid variable is not defined properly."')


# ## 12. Finding the best performing model
# <p>We have defined the grid of hyperparameter values and converted them into a single dictionary format which <code>GridSearchCV()</code> expects as one of its parameters. Now, we will begin the grid search to see which values perform best.</p>
# <p>We will instantiate <code>GridSearchCV()</code> with our earlier <code>logreg</code> model with all the data we have. We will also instruct <code>GridSearchCV()</code> to perform a <a href="https://www.dataschool.io/machine-learning-with-scikit-learn/">cross-validation</a> of five folds.</p>
# <p>We'll end the notebook by storing the best-achieved score and the respective best parameters.</p>
# <p>While building this credit card predictor, we tackled some of the most widely-known preprocessing steps such as <strong>scaling</strong>, <strong>label encoding</strong>, and <strong>missing value imputation</strong>. We finished with some <strong>machine learning</strong> to predict if a person's application for a credit card would get approved or not given some information about that person.</p>

# In[202]:


# Instantiate GridSearchCV with the required parameters
grid_model = GridSearchCV(estimator=logreg, param_grid=param_grid, cv=5)

# Fit grid_model to the data
grid_model_result = grid_model.fit(rescaledX_train, y_train)

# Summarize results
best_score = grid_model_result.best_score_
best_params = grid_model_result.best_params_
print("Best: %f using %s" % (best_score, best_params))

# Extract the best model and evaluate it on the test set
best_model = grid_model_result.best_estimator_
accuracy = best_model.score(rescaledX_test, y_test)
print("Accuracy of logistic regression classifier: ", accuracy)


# In[203]:


get_ipython().run_cell_magic('nose', '', '\ncorrect_grid_model = GridSearchCV(estimator=logreg, param_grid=param_grid, cv=5)\ncorrect_grid_model_result = correct_grid_model.fit(rescaledX_train, y_train)\n\n\ndef test_grid_model_defined():\n    assert "grid_model" in globals(), "The variable grid_model should be defined."\n\n\ndef test_grid_model_defined_correctly():\n    # correct_grid_model = GridSearchCV(estimator=logreg, param_grid=param_grid, cv=5)\n    assert str(correct_grid_model) == str(\n        grid_model\n    ), "It doesn\'t appear that `grid_model` was defined correctly."\n\n\ndef test_grid_model_results_defined():\n    assert (\n        "grid_model_result" in globals()\n    ), "The variable `grid_model_result` should be defined."\n\n\ndef test_grid_model_result_defined_correctly():\n    #     correct_grid_model = GridSearchCV(estimator=logreg, param_grid=param_grid, cv=5)\n    #     correct_grid_model_result = correct_grid_model.fit(rescaledX, y)\n    assert str(correct_grid_model_result) == str(\n        grid_model_result\n    ), "It doesn\'t appear that `grid_model_result` was defined correctly."\n\n\ndef test_best_score_defined_correctly():\n    #     correct_grid_model = GridSearchCV(estimator=logreg, param_grid=param_grid, cv=5)\n    #     correct_grid_model_result = correct_grid_model.fit(rescaledX, y)\n    correct_best_score = correct_grid_model_result.best_score_\n    assert (\n        correct_best_score == best_score\n    ), "It looks like the variable `best_score` is not defined correctly."\n\n\ndef test_best_params_defined_correctly():\n    #     correct_grid_model = GridSearchCV(estimator=logreg, param_grid=param_grid, cv=5)\n    #     correct_grid_model_result = correct_grid_model.fit(rescaledX, y)\n    correct_best_params = correct_grid_model_result.best_params_\n    assert (\n        correct_best_params == best_params\n    ), "It looks like the variable `best_params` is not defined correctly."\n\n\ndef test_best_model_defined_correctly():\n    correct_best_model = correct_grid_model_result.best_estimator_\n    assert (\n        str(correct_best_model) == str(best_model)\n    ), "It looks like the variable `best_model` is not defined correctly."')

