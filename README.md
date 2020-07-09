# nationwide_modeling_assignment
Task: Cleaning and modeling data while investigating the performance capabilities of the Logistic Regression and Xtreme Gradient Boost classification algorithms.

  My modeling process began by importing the supplied data files to separate dataframes and checking that all ids are unique such that all instances of data 
were separate and uniquely linked to an id. Having found that all ids were indeed unique throughout each of the data files provided, I proceeded to assess 
the dataset for patterns in the missing data. Using the missingno library and its matrix function, I saw no discernable patterns of missing data throughout 
the groupAlearning or groupBlearning columns. Based on this observation, I did not remove any rows or columns of the learning dataset due to patterns of missing 
data. From this point, I merged the pairs of learning and prediction datasets with a join function targeting the id instances. I proceeded to check the column 
types contained in the learning dataset. I found that the dataset represented object, float, and integer data types. I chose to investigate which columns were of 
type object and found that columns ‘b00’, ‘b34’, ‘b35’, and ‘b58’ would require some formatting to standardize their respective data entries. With a cleaned 
dataset, I used the fillna function to impute all of the null instances contained in numerical columns with their column mean. I then used the get_dummies 
function from the pandas library to convert the categorical columns to a one-hot encoded dataframe that replaced the categorical columns in the learning dataset. 
I then assessed the learning dataset for multicollinearity between its features and found that there were three pairs of features correlated above a threshold 
magnitude of .5. Using a function that returns the correlation measurement of each of these features to the target class, I made the decision to drop the 
‘b34_wednesday’ and ‘b21_Central’ columns. 

  I chose the logistic regression algorithm and the extreme gradient boost classifier to compare in this exercise. In practice, I have found the logistic 
regression algorithm to handle classification tasks with diverse training data very well, in part due to regularization options, but also with respect to the 
built in class_weight parameter that become very useful to attain useful results while training the model on an imbalanced dataset. I chose the extreme gradient 
boost algorithm for its well known capabilities to classify diverse dataset, as well as its class weighting parameter, scale_pos_weigh. 
Using the GridSearch method I found a set of suitable parameters that returned area under the curve scores of 86.92% on the logistic model and 95.27% on the 
XBGC model for the training set. Applying the testing set, the logistic regression returned an AUC score of 91.1% and the extreme gradient boost classifierl 
returned a score of 100%. While I did expect the XGBC algorithm to perform well, I did not expect to see these results. In past experience, such a high scoring 
metric would undoubtedly reflect an amount of data leakage in the training process. To test this, I modeled a dataset without filling the null values in the 
numerical columns with their column mean. I found that the XGBC algorithm returns an equally high recall and precision score on either training and testing set 
as the scores of the XGBC model on the dataset that had its values imputed with their column mean. In this way, I conclude that mean-wise imputation did not 
impart an undue effect on the XGBC’s score.

  Looking towards model generalizability, we see in the classification report reflects that the logistic regression model had a .04 difference in recall score on 
the positive case between training and testing sets, as well as a .04 difference in precision score on the positive case between training and testing sets. 
Because of the low margin of difference between these scores, It would be expected to see similar results when new data is introduced. Though more regularization 
parameter tuning may be necessary as a larger quantity of data is introduced as well as further adjustment to help maximize the low precision scores on the 
positive case. Evaluating the XGBC model we see very high scores again, as the classification report reflects no difference between the testing and training 
datasets. In this way, we would expect the model to generalize very well as new data is introduced, with minor to no adjustment of regularization parameters 
and class weight measurements. 

  While ROC-AUC scores are a key identifier of a classification model’s performance, If the exercise had not imposed the metric, I find the recall and precision 
scores to be just as useful to interpret a model’s significance. 
Considering changes to the scale of data, If the dataset were to increase, I would expect to change the regularization parameter of the logistic regression model 
to L1 as the penalty function works to decrease unimportant features to a zero coefficient, inherently acting as a feature selection tool. I would also expect to 
see similar changes in regularization parameters in the L1 and L2 lambda coefficients in the XGBC model. If the dataset was decreased by a significant amount, 
it may be useful to disable the regularization parameters completely to keep from overfitting the data. In either case, the class weight parameters would need 
to be adjusted to account for the changes in target class ratio.

  If data augmentation had been permitted, I would have liked to incorporate data that reflected the year and time of recording of each transaction instance. 
Although we have zipcode data, introducing more specific location data like city or address could prove useful. 
If I were given another week to work on this exercise, I would try to apply different linear algorithms such as the support vector classifier. And further work 
on predictive performance would lead to more precise and incremental tuning of the current and additional hyper-parameters as well as a much more extensive 
resampling approach to cross-validation during the training processes. 
Looking at the AUC, recall, and precision metrics we see that the extreme gradient boost classifier performed best overall as it offers a wider range of parameter 
tuning capabilities. Although the XGBC algorithms have been known to overfit data, this particular model instance is expected to generalize better than 
the logistic model on this particular set of data. 

  Although not the case in this instance, logistic regression does suffer from a weakness known as complete separation of features. This weakness is exploited 
when the features of the data completely separate the target classes. In terms of computational cost and implementation complexity, the logistic regression 
is much quicker and simple to implement and returns a robust and consistent model in very little time. 
