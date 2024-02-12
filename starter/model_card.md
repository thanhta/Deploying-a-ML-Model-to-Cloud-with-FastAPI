# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
* Developed by Thanh Ta, Feburary 2024
* Version 1.0
* Prediction task is to determine if a person earns over 50K per year or not. We use Random Forest classfier for the prediction task. The parameters to be used are default values.

## Intended Use
The intended use is to predict a person earns over 50K per year or not based on the public US Census Incomde data from [UCI Machine Learning Repository]( https://archive.ics.uci.edu/dataset/20/census+income ). One of the application usage is to obtain loan approval based on the person's predicted income.

## Training Data
The source of the data for training is from [UCI Machine Learning Repository]( https://archive.ics.uci.edu/dataset/20/census+income ). 80% of the data is used for training. 

The below columns are used as features. These categorical features are encoded to numerical values using "OneHotEncoder":
* workclass
* education
* marital-status
* occupation
* relationship
* race
* sex
* native-country

The target label is "salary". It has been transformed to numerical values using "LabelBinarizer":
* ">50K" as 1
* "<=50k" as 0

## Evaluation Data
The source of the data for evaluation is from [UCI Machine Learning Repository]( https://archive.ics.uci.edu/dataset/20/census+income ). 20% of the data is used for validation.

## Metrics
The below metrics are used to evaluate the performance of the Random Forest classfier model:
* Precision: The number of true positives divided by the total number of positive predictions (i.e., the number of true positives plus the number of false positives). In this case, the Precision value is 0.73 
* Recall: The number of true positive predictions divided by the total actual positives. In this case, the Recall value is 0.64 
* F-beta: The weight harmonic of precision and recall. In this case the F-beta value is 0.68

## Ethical Considerations
This model is trained on the public US Census Income data. This data set has a bias towards men (twice as many men as women) and it also has a bias towards white people in the US.

## Caveats and Recommendations
Since this US Census Income data was extracted by Barry Becker from the 1994 Census database, it is an outdated sample and should not be considered as a fair representation of the salary distribution. This 
dataset should be used only for training purpose on Machine Learning classification.