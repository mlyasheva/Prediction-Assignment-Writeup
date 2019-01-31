---
title: "Practical Machine Learning - Prediction Assignment Writeup"
author: "Maria Lyasheva"
output: html_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```



# Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

# Aim

Apply a machine learning algorithm to the 20 test cases available in the test data above and submit the predictions in appropriate format to the Course Project Prediction Quiz for automated grading.

# Data loading

The pml_training and pml_testing datasets were downloaded from the following websites:
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv
Original data can be found on the following website:
http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har

# The packages, reqired for the analysis, were loaded.

```{r}

library(lattice)
library(ggplot2)
library(caret)
library(rpart)
library(rpart.plot)
library(corrplot)
library(rattle)
library(randomForest)
library(RColorBrewer)
```

The dateset pml_training were read and the information about this data set was obtained.

```{r}
setwd("~/Desktop/Coursera Machine Learning")
url_train <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
data_train <- read.csv(url(url_train), strip.white = TRUE, na.strings = c("NA",""))
View(data_train)
dim(data_train)
```


Similarly, the dateset pml_testing was read and information about this data set was extracted.

```{r}
url_quiz  <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
data_quiz  <- read.csv(url(url_quiz),  strip.white = TRUE, na.strings = c("NA",""))
View(data_quiz)
dim(data_quiz)
```


Both datasets were found to have the same number of variables - 160.

# Data Loading and Cleaning
Data Partitioning:
To do a data partitioning two partitions (75 % and 25 %) within the original training dataset (data_train) were created. This was used for cross validation purpose.

```{r}
in_train  <- createDataPartition(data_train$classe, p=0.75, list=FALSE)
train_set <- data_train[ in_train, ]
test_set  <- data_train[-in_train, ]

dim(train_set)
dim(test_set)
```

There were many NA values and near-zero-variance variables in two mentioned datasets (train_set and test_set). Thus, these values and variables as well as ID variables were be removed for the further analysis.

```{r}
nzv_var <- nearZeroVar(train_set)
train_set <- train_set[ , -nzv_var]
test_set  <- test_set [ , -nzv_var]
dim(train_set)
dim(test_set)
```

Folowing this, the variables that were close to NA were excluded via selecting a threshlod of 95%.

```{r}
na_var <- sapply(train_set, function(x) mean(is.na(x))) > 0.95
train_set <- train_set[ , na_var == FALSE]
test_set  <- test_set [ , na_var == FALSE]

dim(train_set)
dim(test_set)
```

Lastly, colums 1, 2, 3, 4 and 5 were also removed from the datasets because they contained only identification variables.

```{r}
train_set <- train_set[ , -(1:5)]
test_set  <- test_set [ , -(1:5)]

dim(train_set)
dim(test_set)
```

As a result, the original number of variables (160) was reduced to 54.

# Correlation Analysis

A correlation analysis was useful to highlight how much variables were correlated between each other.
Thus, this analysis was performed before the modeling work itself.
"Hclust" order was selected to organise the data.
Figure 1: Correlation matrix visualization

```{r}
corr_matrix <- cor(train_set[ , -54])
corrplot(corr_matrix, order = "hclust", method = "square", type = "lower", tl.cex = 0.6, tl.col = rgb(0, 0, 0))
```

The correlation coefficients were coloured according to the value (red is =-1 (negative corraltions), blue is =1 (positive correlation)) so it was easier to understand what was correlating with what and how much.
A Principal Components Analysis (PCA) could be further applied in order to reduce the number of variables but there were not many variables that had strong correlations and, thus, PCA was not applied.

# Prediction Models

The models that were probed in this coursework were Decision Tree Model, Generalized Boosted Model (GBM) and Random Forest Model.
This probing helped to choose the best prediction model.

## 1. Decision Tree Model

```{r}
set.seed(1834)
fit_decision_tree <- rpart(classe ~ ., data = train_set, method="class")
```

Figure 2: Decision Tree Plot 

```{r}
fancyRpartPlot(fit_decision_tree)
```

Predictions of the decision tree model on test_set.

```{r}
predict_decision_tree <- predict(fit_decision_tree, newdata = test_set, type="class")
conf_matrix_decision_tree <- confusionMatrix(predict_decision_tree, test_set$classe)
conf_matrix_decision_tree
```

The predictive accuracy of the Decision Tree Model was low, 72.92%.

## 2. GBM

```{r}
set.seed(1834)
ctrl_GBM <- trainControl(method = "repeatedcv", number = 5, repeats = 2)
fit_GBM  <- train(classe ~ ., data = train_set, method = "gbm",
                  trControl = ctrl_GBM, verbose = FALSE)
fit_GBM$finalModel
```

Predictions of the GBM on test_set.

```{r}
predict_GBM <- predict(fit_GBM, newdata = test_set)
conf_matrix_GBM <- confusionMatrix(predict_GBM, test_set$classe)
conf_matrix_GBM
```

The predictive accuracy of the GBM was high, 98.63%.

## 3. Random Forest Model

```{r}
set.seed(1834)
ctrl_RF <- trainControl(method = "repeatedcv", number = 5, repeats = 2)
fit_RF  <- train(classe ~ ., data = train_set, method = "rf",
                 trControl = ctrl_RF, verbose = FALSE)
fit_RF$finalModel
```

Predictions of the Random Forest Model on test_set.

```{r}
predict_RF <- predict(fit_RF, newdata = test_set)
conf_matrix_RF <- confusionMatrix(predict_RF, test_set$classe)
conf_matrix_RF
```

The predictive accuracy of the Random Forest Model was very high, 99.82%.

# Applying the Best Predictive Model to the Test Data

To summarise, Decision Tree Model had the lowest accuracy (72.92%), whereas Generalized Boosted Model and Random Forest Model had higher accuracy, 98.63% and 99.82% respectively. The Random Forest model was found to be the most accurate model and was selected and applied to make predictions on the 20 data points from the original testing dataset (data_quiz).

```{r}
predict_quiz <- predict(fit_RF, newdata = data_quiz)
predict_quiz
```

