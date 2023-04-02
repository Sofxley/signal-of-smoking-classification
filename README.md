# Signal of Smoking Classification: Project Overview

<a href="https://www.kaggle.com/code/blulypsee/signal-of-smoking-classification-77-34"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open in Kaggle" /></a> 

I recommend reading the project notebook on Kaggle in case you can not view it on Github.

## SUMMARY

* Built eight different classification models to predict whether the person is a smoker, resulting in the best model being Random Forest with accuracy over 77%, recall equal to 0.73 and precision of 0.68.
* Conducted exploratory data analysis on regular health check-ups data to find the factors such as hemoglobin level and triglyceride level that showed strong indication of a smoking habit. Also, discovered that there is no obvious relationship between certain features, f.e. HDL and LDL levels to gender and smoking habit, which is unique to the dataset used.
* Established a baseline level of performance of a set of various models using different variations of the dataset received after scaling the data, removing the outliers or reducing the dimensionality of the dataset.
* Optimized Logistic Regression, Stochastic Gradient Descent, K-Nearest Neighbors, CART, AdaBoost, Random Forest, Extra Trees, Bagging Classifier using GridSearchCV, Threshold-Moving for Imbalanced Classification and Recursive Feature Elimination technique to reach the best model. 

## RESULTS

In this project I have been trying to build a model that correctly classifies smokers and non-smokers using the data from regular health check-ups. I have made an attempt at fine-tuning a couple of classifiers presented in the table below. 

![model comparison](https://github.com/Sofxley/signal-of-smoking-classification/blob/main/images/model_comparison.png)

Unfortunately, we did not manage cross the accuracy threshold of 80% (as for this version of notebook),  **the highest accuracy that was reached is 77.34% using Random Forest model at 0.732 recall and 0.677 precision.**  This model turns out to perform better than any other in the table, though all the accuracies actually are roughly in the range from 73% to 78%. That said,  **Random Forest**  outperforms any other model, followed right after by Extra Trees, KNN, and Bagging Classifier.

The performance of Random Forest on the test set is shown using the following confusion matrix.

![rf confusion mx for test set](https://github.com/Sofxley/signal-of-smoking-classification/blob/main/images/rf_matrix_results.png)

We can see that the optimized Random Forest model classifies 52.63% out of 63.27% of non-smokers correctly and only 8.92% out of 36.73% smokers the model predicts to be non-smokers when actually they are not.

## The Outline

 1. [Project At A Glance](#Project_At_A_Glance)
 2. [Introduction](#Intro)
 3. [Problem Statement and Motivation](#Problem_Motiv)
 4. [EDA](#EDA)
 5. [Spot-Checking and Building the Models](#models)
 6. [Limitations and Future Work](#limit_future)

<a id="Project_At_A_Glance"></a>
## Project At A Glance

__Python Version__: 3.7.12

__Packages__: numpy, pandas, matplotlib, seaborn, plotly, sklearn.

<a id="Intro"></a>
## Introduction
The reason for correctly classifying smokers and non-smokers generally is to better understand the relationship between smoking and various health outcomes. Smoking is a significant risk factor for a range of health conditions, including lung cancer, heart disease, stroke, respiratory diseases, and many others. Accurately identifying smokers and non-smokers allows researchers to study the impact of smoking on health outcomes in a more accurate and precise manner. This, in turn, can help guide public health interventions aimed at reducing smoking rates and preventing related health problems.

In addition, correctly classifying smokers and non-smokers can also have practical implications. For example, insurance companies may charge higher premiums to smokers due to their increased risk of health problems, and employers may choose not to hire smokers due to concerns about productivity etc.

It is generally more important to correctly identify smokers than non-smokers, because the health risks associated with smoking are much greater than the risks associated with not smoking. However, it is also important to minimize the number of false alarms (misclassifying non-smokers as smokers). False alarms can have negative consequences, such as stigmatizing individuals who do not smoke or causing unnecessary anxiety.

Therefore, the ideal approach would be to aim for high accuracy in both identifying smokers and non-smokers, while also minimizing false alarms. This can be achieved through the use of machine learning algorithms that are optimized to balance recall (identifying true smokers) and specificity (avoiding false alarms).

<a id="Problem_Motiv"></a>
## Problem Statement and Motivation

I believe the reason why I picked this particular dataset was the fact that I was genuinely interested in what it is about the person that reveals whether they are a smoker or not with so many smokers around nowadays. Throughout the whole process of exploring the dataset I was questioning myself whether the characteristics that caught my eye would make sense in general, not just for a  given dataset. It turned out that some of them are not quite representative since we have an unbalanced dataset and we can not really tell whether the dataset is to blame or is it something unique to this particular population of people or, hopefully, something even more general that points to much deeper relationships than these that we know of. Yet, fortunately, most of the signals are in accord with what we know and do make sense. The end goal was to build a good model, which recognizes the habit of smoking in people. 

The outline of my work looked roughly like this.
First of all, it was exploring the data using visualization tools, checking its sanity and drawing conclusions from the plots. Then, I performed the spot-checking to get a grasp of what type of classification models might be a good match for the problem at hand. Finally, I did some fine-tuning for chosen models and compared them at the very end of the notebook. 

<a id="EDA"></a>
## EDA
Before actually getting down to building the models, I have looked at different distributions of the features while seeking the relationships between them. Below you can see a few highlights from EDA.

First thing, of course, is the correlation plot of all features. 
![corr_plot](https://github.com/Sofxley/signal-of-smoking-classification/blob/main/images/correlation_plot.png)

We notice that a lot of features are highly correlated with each other, for example `alt` with `ast` , `gender` with `height` and `hemoglobin`, or `ldl` with `cholesterol`. These features can either benefit our models or, quite the opposite, hurt them because they may not bring any additional information to the models, but only increase the complexity of the algorithm, thus leading to overfitting.

Moreover, some of the correlations do make sense, e.g. commonly the greater the weight is, the higher is the level of hemoglobin, or the greater the weight is, the lower is the HDL.  Also, the level of hemoglobin depends on a gender, and usually men tend to have a higher level.

![hemoglobin w.r.t. gender](https://github.com/Sofxley/signal-of-smoking-classification/blob/main/images/hemoglobin_wrt_gender.png)

As I've already mentioned, men and women have different mean hemoglobin levels - women have mean levels approximately 12% lower than men. And we see the apparent difference between the curves in the plot that confirms the statement. Furthermore, a higher than average hemoglobin level occurs most commonly when the body requires an increased oxygen-carrying capacity usually due to the smoking habit, living at a high altitude (red blood cell production naturally increases to compensate for the lower oxygen supply there) or other reasons. So we have to take into account that not only the smoking habit contributes to the higher hemoglobin levels when drawing conclusions.

![hemoglobin w.r.t to smoking](https://github.com/Sofxley/signal-of-smoking-classification/blob/main/images/hemoglobin_wrt_smoking.png)

Here again we see the distribution of hemoglobin, and the plot actually shows us that smokers have higher median hemoglobin levels than non-smokers, as well as, it tells us that males have higher hemoglobin levels than females, as we expected it to be.

![violin plots of hdl and ldl](https://github.com/Sofxley/signal-of-smoking-classification/blob/main/images/violin_plot_hdl_ldl.png)

__However, sometimes data does not live up to our expectations.__ So if we were to take a close look at the violin plots above, after doing some analysis we would come to the conclusion that either something is off about our particular dataset or, what's more exciting, there's something wrong about our general understanding of the topic (but you have to check it more than twice before finally drawing a conclusion like that).

Let me explain it to you. Commonly, smokers have lower levels of the “good” cholesterol HDL and higher levels of the “bad” cholesterol LDL. Besides, taking the ages from 20 to 55 years, men tend to have higher LDL and lower HDL cholesterol levels than women. Also, after the age of 55, women's HDL cholesterol levels decrease rapidly, and LDL levels rise. 

Most of the females in our dataset are in the age of 40 and older, while the age range for males is wider and is from 20 to 80. And so we see that for men and women HDL levels tend to get only a bit lower if they are smoking. As for LDL, there is an apparent difference in levels for females: we see that smoking females, contarary to statistics, tend to have lower LDL in our dataset than these not smoking. Besides, the same can be said about the plot for men, though the difference between the levels is not really visible there. In general, if we compare the levels for women and men separately without considering smoking habits, we see that indeed men have lower HDL levels than women (since the age range is wider), but not necessarily the higher LDL, probably because the dataset contains adult females older than 40 and they are known to have rises in LDL in that age.

Although I must mention that I am __not__ an expert in the field of medicine or health, I would take these results as the hint to get your hands on a much __larger__ dataset (perhaps even the [original](https://www.data.go.kr/data/15007122/fileData.do) one for the [dataset used](https://www.kaggle.com/datasets/kukuroo3/body-signal-of-smoking)) and explore these relationships more thoroughly and  try to explain the results you get. You never know whether you will discover something new that will bring us closer to understanding how the smoking habit influences our bodies.

<a id="models"></a>
## Spot-Checking and Building Models

In the 'Spot-Checking' section I have established a baseline for the expected level of performance for a set of models, most of which later on were used in the actual 'Building Models' section.

To sum up, I have concluded that linear models usually require scaling to fit the data well, whilst for other models it was not something particularly necessary. The models' scores commonly stay the same if you try deleting highly correlated features or, what's more, the scores get worse. If we were to decrease the number of features of the data passed to the algorithm using the feature importances technique, then we would not see any improvement too. However, there might be hope that the right outlier detection technique might make the scores go up a bit. 

Next, I have explored the ways I can bring each of the models of interest to their full potential. I was fine-tuning the models and analyzing their performances using common binary classification metrics, such as confusion matrix, precision, recall and the ROC AUC score. I have also made an attempt at boosting their performance by using Recursive Feature Elimination technique and the adjustment of the classification threshold if possible.

<a id="limit_future"></a>
## Limitations and Future Work

First of all, it should be mentioned that, as for any work of mine, there is a lot that could have been done differently and thus would have given much better results.  In this project, some steps are obviously not shown to save the reader's time and also to decrease the runtime of this notebook. Due to the same reasons I restrained myself from using greater numbers for `n_estimators` parameter to get better results when I was sure I could or when choosing the base estimators for RFE and so forth. I haven't explored every possible technique for increasing the accuracy of any model in this project, because it was not essentially my goal. Of course, other techniques could be used for the model of interest, but I will leave the explanation and the use of these methods to other publications/notebooks/etc. I was trying to find out which models should be chosen to be further on explored for this particular dataset, not to focus on any particular model.

Basically, this work is still in progress since I have not yet touched more advanced models or made use of outlier detection technique mentioned in spot-checking at all. Also, the notebook could be made more concise. That aside, let me give you a quick overview of things that I was planning to do:

-   Check the performance of more advanced models such as Boosting/Stacking, XGBoost, Light GBM, NN.
-   Explore outlier detection techniques.
-   Explore more feature selection techniques.
-   and more..?
