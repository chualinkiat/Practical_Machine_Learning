---
title: "Practical Machine Learning Training"
author: "Chua Lin Kiat"
output: 
  html_document:
  keep_md : yes
  toc: yes

---



```
## Run time: 2016-06-19 15:33:55
## R version: R version 3.2.5 (2016-04-14)
```

## Practical Machine Learning Training


## Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

## What you should submit

The goal of your project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. You may use any of the other variables to predict with. You should create a report describing how you built your model, how you used cross validation, what you think the expected out of sample error is, and why you made the choices you did. You will also use your prediction model to predict 20 different test cases.

## Data Processing

Load the csv file pml-training.csv and pml-testing.csv into R 


```r
training <- read.csv("pml-training.csv", na.strings=c("NA","#DIV/0!",""))
testing <- read.csv("pml-testing.csv", na.strings=c("NA","#DIV/0!",""))
```

Load necessary library


```r
library(caret)
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```r
library(rpart)
library(rpart.plot)
library(RColorBrewer)

library(randomForest)
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

# Cross-validation

Cross-validation will be performed by subsampling our training data set into 2 subsamples: subTraining data (60% of the original Training data set) and subTesting data (40%). The models will be fitted on the subTraining data set, and tested on the subTesting data. Once the most accurate model is choosen, it will be tested on the original Testing data set.

# Subsampling the data

Partioning  data set into two data sets, 60% for myTraining, 40% for myTesting:


```r
inTrain <- createDataPartition( y=training$classe , p=0.6, list=FALSE)
myTraining <- training[inTrain, ]; myTesting <- training[-inTrain, ]
dim(myTraining)
```

```
## [1] 11776   160
```

```r
dim(myTesting)
```

```
## [1] 7846  160
```

# Cleaning the training data set

Steps to clean the dataset:-

1.Remove the first 7 columns because do not want it to interfere with the algorithm


```r
myTraining <- myTraining[,-c(1:7)]
myTesting <- myTesting[,-c(1:7)]
```

2.Remove columns with many null or zero values


```r
myTraining <- myTraining[,colSums(is.na(myTraining)) == 0]

myTesting <- myTesting[,colSums(is.na(myTesting)) == 0]
```

Lets see the dimension of the 2 dataset


```r
dim(myTraining)
```

```
## [1] 11776    53
```

```r
dim(myTesting)
```

```
## [1] 7846   53
```

# Deciding the best prediction model

After we have clearn the dataset, we will begin to decide which is the best model. We will investigate the 2 prediction model which is decision tree and random forests

1. accuracy of decision tree prediction
---------------------------------------

Using confusionmatrix to test the decision tree prediction model


```r
modFitA1 <- rpart(classe ~ ., data=myTraining, method="class")
rpart.plot(modFitA1, main="Classification Tree", extra=102, under=TRUE, faclen=0)
```

<img src="PML_files/figure-html/unnamed-chunk-8-1.png" width="672" />

```r
predictionsA1 <- predict(modFitA1, myTesting, type = "class")
confusionMatrix(predictionsA1, myTesting$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2086  326  100  223   57
##          B   45  848   76   31   82
##          C   39  205 1095  164  202
##          D   47  109   95  731   75
##          E   15   30    2  137 1026
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7374          
##                  95% CI : (0.7276, 0.7472)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.665           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9346   0.5586   0.8004  0.56843   0.7115
## Specificity            0.8742   0.9630   0.9058  0.95030   0.9713
## Pos Pred Value         0.7471   0.7837   0.6422  0.69158   0.8479
## Neg Pred Value         0.9711   0.9009   0.9555  0.91825   0.9373
## Prevalence             0.2845   0.1935   0.1744  0.16391   0.1838
## Detection Rate         0.2659   0.1081   0.1396  0.09317   0.1308
## Detection Prevalence   0.3559   0.1379   0.2173  0.13472   0.1542
## Balanced Accuracy      0.9044   0.7608   0.8531  0.75937   0.8414
```

Decision tree accuracy = 0.7369

2. accuracy of random forests prediction
---------------------------------------

Using confusionmatrix to test the random forests prediction model


```r
modFitB1 <- randomForest(classe ~. , data=myTraining)
predictionsB1 <- predict(modFitB1, myTesting, type = "class")
confusionMatrix(predictionsB1, myTesting$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2231   17    0    0    0
##          B    1 1499   10    0    0
##          C    0    2 1356   17    3
##          D    0    0    2 1268    0
##          E    0    0    0    1 1439
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9932          
##                  95% CI : (0.9912, 0.9949)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9915          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9996   0.9875   0.9912   0.9860   0.9979
## Specificity            0.9970   0.9983   0.9966   0.9997   0.9998
## Pos Pred Value         0.9924   0.9927   0.9840   0.9984   0.9993
## Neg Pred Value         0.9998   0.9970   0.9981   0.9973   0.9995
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2843   0.1911   0.1728   0.1616   0.1834
## Detection Prevalence   0.2865   0.1925   0.1756   0.1619   0.1835
## Balanced Accuracy      0.9983   0.9929   0.9939   0.9928   0.9989
```

Random forests accuracy = 0.9935

Conclusion: The random forests model is more accurate as it provide an accuracy nearly 100% while decision tree gives a worse accuracy.

# Submission

Generate the submission file based on random forests prediction



```r
predictionsB2 <- predict(modFitB1, testing, type = "class")
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(predictionsB2)
```


#---------------------------------------------------------


