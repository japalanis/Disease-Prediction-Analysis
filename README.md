# Disease-Prediction-Analysis
In this report we will be analysing the Disease prediction dataset.We are going to create various disease diagnosis models using Logistic Regression, Decision Tree, Artificial Neural Network in order to predict whether the patient has the specific disease.

#### **Package Loading:**

Load all the required packages:

```{r,results='hide', message=FALSE, warning=FALSE}
require(MASS)
library(ggplot2)
library(dplyr)
library(tidyr)
library(tidyverse)
library(klaR)
library(e1071)
library(mlbench)
library(kernlab)
library(recipes)
library(pROC)
library(precrec)
library(keras)
library(tensorflow)
library(yardstick)
library(caret)
library(ROCR)
library(data.table)
```

#### **Data Loading:**

Let us read the training and testing datasets.

```{r}
cardio_train<-read.csv("C:/Users/ragha/Desktop/Priya/Disease Prediction Training.csv", header=TRUE)
cardio_test<-read.csv("C:/Users/ragha/Desktop/Priya/Disease Prediction Testing.csv", header=TRUE)
```

#### **Data Exploration:**

```{r}
str(cardio_train)
```

```{r}
summary(cardio_train)
```

NA values:

```{r}
sapply(cardio_train, function(x) sum(is.na(x)))
```

Since there are no NA's in the dataset, we can move further with our analysis.

Removing the duplicate values from the dataset:

```{r}
cardio_train<-cardio_train[!duplicated(cardio_train),]
```

Handling the Outliers & Data Visualization:


```{r}
ggplot(cardio_train,aes(x = Gender))+
  geom_bar(fill="orange",color="brown")+
  xlab("Gender")+
  ylab("Count")+
  ggtitle("Gender Distribution")
```


```{r}
boxplot(cardio_train$High.Blood.Pressure,
main = "High Blood Pressure Distribution",
ylab="High BP",
border = "brown",
vertical = TRUE
)
```

```{r}
boxplot(cardio_train$Low.Blood.Pressure,
main = "Low Blood Pressure Distribution",
ylab="Low BP",
border = "brown",
vertical = TRUE
)
```


From the above charts,there were many outliers in Low BP and High BP columns with values ranging from negatives to several thousands. I have removed the rows of the Low BP and High BP values that makes no sense by limiting it to particular range.I have taken the Low BP range from 20-190 and High BP range from 60-240.

```{r}
cardio_train <- cardio_train[cardio_train$Low.Blood.Pressure >= 20,]
cardio_train <- cardio_train[cardio_train$Low.Blood.Pressure <= 190,]

cardio_train <- cardio_train[cardio_train$High.Blood.Pressure>= 60,] 
cardio_train <- cardio_train[cardio_train$High.Blood.Pressure <= 240,]

```


From the above charts,there were many outliers in Low BP and High BP columns with values ranging from negatives to several thousands. I have removed the rows of the Low BP and High BP values that makes no sense by limiting it to particular range.I have taken the Low BP range from 20-190 and High BP range from 60-240.

```{r}
cardio_train <-  cardio_train[cardio_train$Height >= 100,]
cardio_train <-  cardio_train[cardio_train$Weight >= 20,]

```

Converting the values to numeric data type:

```{r}
cardio_train$Age<-as.numeric(cardio_train$Age)
cardio_train$Height<-as.numeric(cardio_train$Height)
cardio_train$Weight<-as.numeric(cardio_train$Weight)
cardio_train$Low.Blood.Pressure<-as.numeric(cardio_train$Low.Blood.Pressure)
cardio_train$High.Blood.Pressure<-as.numeric(cardio_train$High.Blood.Pressure)
cardio_train$Smoke<-as.numeric(cardio_train$Smoke)
cardio_train$Alcohol<-as.numeric(cardio_train$Alcohol)
cardio_train$Exercise<-as.numeric(cardio_train$Exercise)
cardio_train$Disease <- as.factor(cardio_train$Disease)
```

#### **Training and Testing Dataset:**

Let us now split this data into train & test:

```{r}
train_index <- createDataPartition(cardio_train$Disease, p = 0.8, list = FALSE)
data_train <- cardio_train[train_index, ]
data_test <- cardio_train[-train_index, ]
```


#### **Logistic Regression:**

Logistic regression is a basic model that uses a logistic function to model a binary dependent variable.Logistic regression is used in various fields, including machine learning, most medical fields, and social sciences.

Creating the model using the training dataset:

```{r,results='hide', message=FALSE, warning=FALSE}
start1 <- Sys.time()
model_glm <- train(Disease ~ ., data = data_train, 
                   method = "glm", family = "binomial")
Sys.time() - start1
```

```{r}
print(model_glm)
```

Lets see the summary of the model:

```{r}
summary(model_glm)
```

Now,lets plot the ROC curve for the model

```{r message=F, warning=F}
model_glm_prob <- predict(model_glm, newdata = data_test, type = "prob")
roc_curve_glm1 <- roc(data_test$Disease, model_glm_prob[,1])
roc_curve_glm2 <- roc(data_test$Disease, model_glm_prob[,2])
plot(roc_curve_glm1, col = "red", main = "LR ROC")
plot(roc_curve_glm2, col = "blue", add = T)
legend(0.35, 0.2, legend = c("No", "Yes"), lty = 1,
       col=c("red", "blue"), bty = "n")
```


Variable importance of the model:

```{r, message=FALSE, warning=FALSE}
varImp(model_glm)
```

Let us now go ahead and predict the Disease variable using the above model

```{r}
predictedlr <- predict(model_glm, data_test) 
confusionMatrix(predictedlr, data_test$Disease)
head(predictedlr)
```


#### **Articificial Neural Network:**
An Artificial Neural Network is an information processing model.It is a simple mathematical model of the brain which is used to process nonlinear relationships between inputs and outputs in parallel like a human brain does every second.

Data Preprocessing:

```{r}
rec_obj <- recipe(Disease ~ ., data = data_train) %>%
  step_dummy(all_nominal(), -all_outcomes()) %>%
   step_center(all_predictors(), -all_outcomes()) %>% 
  step_scale(all_predictors(), -all_outcomes()) %>% 
  prep(data = data_train)
rec_obj
```

Splitting the dependent and the independent variables in the training and testing datasets

```{r}
x_train_tbl <- as.matrix(bake(rec_obj, new_data = data_train) %>% select(-Disease))
x_test_tbl <- as.matrix(bake(rec_obj, new_data = data_test) %>% select(-Disease))
y_train_vec <- ifelse(pull(data_train, Disease) == "1", 1, 0)
y_test_vec <-  ifelse(pull(data_test, Disease) == "1", 1, 0)
str(x_train_tbl)
```


First lets create a model with zero hidden layers and then lets see the performance of the model by adding hidden layers.

ANN0- With zero hidden layer:

```{r,results='hide',message=FALSE, warning=FALSE}
start1 <- Sys.time()
model_keras0 <- keras_model_sequential()
model_keras0 %>%
 layer_dense(units = 1, activation = 'sigmoid',input_shape = ncol(x_train_tbl))
head(x_train_tbl)
Sys.time()- start1
```


```{r,results='hide',message=FALSE, warning=FALSE}
start1 <- Sys.time()
model_keras0 %>% compile(
  optimizer = 'adam', 
  loss = 'binary_crossentropy',
  metrics = c('accuracy')
)
Sys.time()- start1
```

```{r}
model_keras0
```

```{r,message=FALSE, warning=FALSE}
start1 <- Sys.time()
set.seed(100)
fit0 <- model_keras0 %>% fit(x_train_tbl,y_train_vec,epochs = 10)
Sys.time()- start1
```


Plot training & validation accuracy values

```{r,message=FALSE, warning=FALSE}
plot(fit0)
```

We can see from the above plot that the accuracy saturates around 72% as the loss continues to drop.

Predicting the Disease:

```{r}
disease_pred0 <- model_keras0 %>% predict_classes(x_test_tbl)
disease_pred0[1:20]

disease_prob0<-model_keras0 %>% predict_proba(x_test_tbl)

```

```{r}
estimates_keras_tbl0 <- tibble(
  truth = as.factor(y_test_vec) %>% fct_recode(yes = "1", no = "0"),
  estimate = as.factor(disease_pred0) %>% fct_recode(yes = "1", no = "0"),
  class_prob = disease_prob0
)
estimates_keras_tbl0
```

Creating the confusion matrix:

```{r}

options(yardstick.event_first = F)
estimates_keras_tbl0 %>% conf_mat(truth, estimate)
```

Calculating the metrics:

```{r}
estimates_keras_tbl0 %>% metrics(truth, estimate)
```

Precision:

```{r}
estimates_keras_tbl0 %>% precision(truth, estimate)
```

Recall:

```{r}
estimates_keras_tbl0 %>% recall(truth, estimate)
```

ANN1- With one hidden layer:

```{r,results='hide',message=FALSE, warning=FALSE}
model_keras1 <- keras_model_sequential()
model_keras1 %>%
  layer_dense(units = 20, activation = 'relu',input_shape = ncol(x_train_tbl)) %>%
  layer_dense(units = 1, activation = 'sigmoid')
head(x_train_tbl)

```

```{r}
model_keras1 %>% compile(
  optimizer = 'adam', 
  loss = 'binary_crossentropy',
  metrics = c('accuracy')
)
```


```{r,message=FALSE, warning=FALSE}
set.seed(100)
fit1 <- model_keras1 %>% fit(x_train_tbl,y_train_vec,epochs = 10)
```

Plot training & validation accuracy values


```{r,message=FALSE, warning=FALSE}
plot(fit1)

```

We can see from the above plot that the accuracy saturates around 73% as the loss continues to drop.

Predicting the Disease:

```{r}
disease_pred1 <- model_keras1 %>% predict_classes(x_test_tbl)
disease_prob1<-model_keras1 %>% predict_proba(x_test_tbl)

```

```{r}
estimates_keras_tbl1 <- tibble(
  truth = as.factor(y_test_vec) %>% fct_recode(yes = "1", no = "0"),
  estimate = as.factor(disease_pred1) %>% fct_recode(yes = "1", no = "0"),
  class_prob = disease_prob1
)
estimates_keras_tbl1
```

Creating the confusion matrix:
```{r}

estimates_keras_tbl1 %>% conf_mat(truth, estimate)
```

Calculating the metrics:

```{r}
estimates_keras_tbl1 %>% metrics(truth, estimate)
```

Precision:

```{r}
estimates_keras_tbl1 %>% precision(truth, estimate)
```

Recall:

```{r}
estimates_keras_tbl1 %>% recall(truth, estimate)
```




ANN2- With two hidden layer:


```{r,results='hide',message=FALSE, warning=FALSE}
model_keras2 <- keras_model_sequential()
model_keras2%>%
  #layer_flatten(input_shape = ncol(rec_obj)) %>%
   layer_dense(units = 50, activation = 'relu',input_shape = ncol(x_train_tbl)) %>%
  layer_dense(units = 20, activation = 'relu') %>%
  layer_dense(units = 1, activation = 'sigmoid')
head(x_train_tbl)
```

```{r}
model_keras2 %>% compile(
  optimizer = 'adam', 
  loss = 'binary_crossentropy',
  metrics = c('accuracy')
)
```


```{r,message=FALSE, warning=FALSE}
set.seed(100)
fit2 <- model_keras2 %>% fit(x_train_tbl,y_train_vec,epochs = 10)
```

Plot training & validation accuracy values

```{r,message=FALSE, warning=FALSE}
plot(fit2)
```

We can see from the above plot that the accuracy saturates around 73% as the loss continues to drop.

Predicting the Disease:

```{r}
disease_pred2 <- model_keras2 %>% predict_classes(x_test_tbl)
disease_prob2<-model_keras2 %>% predict_proba(x_test_tbl)

```

```{r}
estimates_keras_tbl2 <- tibble(
  truth = as.factor(y_test_vec) %>% fct_recode(yes = "1", no = "0"),
  estimate = as.factor(disease_pred2) %>% fct_recode(yes = "1", no = "0"),
  class_prob = disease_prob2
)
estimates_keras_tbl2
```

Creating the confusion matrix:
```{r}
estimates_keras_tbl2 %>% conf_mat(truth, estimate)
```

Calculating the metrics:

```{r}
estimates_keras_tbl2 %>% metrics(truth, estimate)
```

Precision:

```{r}
estimates_keras_tbl2 %>% precision(truth, estimate)
```

Recall:

```{r}
estimates_keras_tbl2 %>% recall(truth, estimate)
```

Comparison between linear SVM, Logistic Regression and ANN0(zero hidden layer):

While comparing these three models, it is found that all three models has similar performance and it just has minute differences in their accuracy. Linear SVM and logistic regression are quite similar because the only difefrence is in the loss function — SVM minimizes hinge loss while logistic regression minimizes logistic loss.And out of the three, Logistic Regression has the highest accuracy of 73% as like its the best model for binary classification problem.


#### **Decision Tree**

Decision Tree algorithm is a supervised learning algorithms. Unlike other supervised learning algorithms, decision tree algorithm can be used for solving regression and classification problems too.

The general motive of using Decision Tree is to create a training model which can use to predict class or value of target variables by learning decision rules inferred from prior data(training data).

Let us now perform decision tree analysis on this training data.

```{r}
dt_model <- train(Disease ~ ., data = data_train, metric = "Accuracy", method = "rpart")
```

```{r}
print(dt_model)
```

```{r}
dt_predict <- predict(dt_model, newdata = data_test, type = "prob")
head(dt_predict, 5)
```

Data Tuning:

```{r message = F, warning = F}
start1 <- Sys.time()
model_dt <- train(Disease ~ ., data = data_train, method = "rpart",
                       metric = "Accuracy",
                       tuneLength = 25,
                       trControl = trainControl(method = "repeatedcv", number = 10, repeats = 3))
Sys.time() - start1
```

Predict the Disease with the tuned model:

```{r}
predict_dt <- predict(model_dt, newdata = data_test, type = "raw")
confusionMatrix(predict_dt, data_test$Disease)
```

Plot the ROC curve of the model:

```{r message=F, warning=F}
dt_pred_prob <- predict(model_dt, newdata = data_test, type = "prob")
roc_curve1 <- roc(data_test$Disease, dt_pred_prob[,1])
roc_curve2 <- roc(data_test$Disease, dt_pred_prob[,2])
plot(roc_curve1, col = "red", main = "Decision Tree ROC")
plot(roc_curve2, col = "blue", add = T)
legend(0.35, 0.2, legend = c("No", "Yes"), lty = 1,
       col=c("red", "blue"), bty = "n")
```

with an accuracy of 73.6%, Decision tree algorithm tops high with better accuracy model than any other model in predicting the disease of a respective patient.

#### **Models for Comparison**

#### **Random Forest**

Random Forest algorithm can be used for both regression and classification problems. Two methods for tuning the model for, 1. Random Search and 2. Grid Search

Grid Search:
In grid search the model is evaluated with all the combinations that are passed in the function, using cross-validation

Random Search:
Unlike grid search, random search will not evaluate all the combinations of hyperparameters, instead a random combination is chosen at every iteration.

For tuning, let us use grid search.

Default setting:

```{r,message=FALSE, warning=FALSE}
trControl <- trainControl(method = "cv",
                          number = 3,
                          search = "grid")
```

```{r,message=FALSE, warning=FALSE}
set.seed(1234)
# Run the model
default_rf_model <- train(Disease~.,
                          data = data_train,
                          method = "rf",
                          metric = "Accuracy",
                          trControl = trControl)
```

Let us test the values of mtry from 1 to 10

```{r,message=FALSE, warning=FALSE}
set.seed(1234)
tuneGrid <- expand.grid(.mtry = c(1:10))
                 
```

Let us train the random forest model with the parameters.

```{r,message=FALSE, warning=FALSE}
start1 <- Sys.time()
fit_rf <- train(Disease~.,
                data_train,
                method = "rf",
                metric = "Accuracy",
                tuneGrid = tuneGrid,
                trControl = trControl,
                importance = TRUE,
                nodesize = 25,
                ntree = 350,
                maxnodes = 30)
Sys.time()-start1

```

Let us predict the disease on test data set

```{r,message=FALSE, warning=FALSE}
prediction <-predict(fit_rf, data_test)
```

Checking for the accuracy

```{r,message=FALSE, warning=FALSE}
confusionMatrix(prediction, data_test$Disease)
```


#### **Support Vector Machine**
A Support Vector Machine (SVM) is a supervised machine learning algorithm that can be employed for both classification and regression purposes. SVMs are more commonly used in classification problems. So, lets go ahead and train the model on the trianing data:

Linear Model:

```{r,message=FALSE, warning=FALSE}
start1 <- Sys.time()
model_svm_linear <- train(Disease ~ ., data = data_train,
                          method = "svmLinear",
                          preProcess = c("center", "scale"),
                          trControl = trainControl(method = "cv", number = 3),
                          tuneGrid = expand.grid(C = seq(0, 1, 0.05)))
Sys.time()-start1

```
Lets have a look at the model:

```{r}
model_svm_linear
```

Lets predict the disease on test data set

```{r,message=FALSE, warning=FALSE}
predict_svm_linear <- predict(model_svm_linear, newdata = data_test)
```

Checking the accuracy of the model

```{r,message=FALSE, warning=FALSE}
confusionMatrix(predict_svm_linear, data_test$Disease)
```

#### **GBM**

GBM is a machine learning technique for regression and classification problems. It builds the model in a stage-wise fashion like other boosting methods do, and it generalizes them by allowing optimization of an arbitrary differentiable loss function.

```{r,message=FALSE, warning=FALSE}
start1 <- Sys.time()

set.seed(825)
gbmFit1 <- train(Disease ~ ., data = data_train, 
                 method = "gbm", 
                 trControl = trainControl(method = "cv", number = 3),
                 verbose = FALSE)
gbmFit1
Sys.time()-start1
```

Lets predict the disease

```{r,message=FALSE, warning=FALSE}
prediction_gbm_tuned <-predict(gbmFit1, data_test)
```

Check for accuracy

```{r,message=FALSE, warning=FALSE}
confusionMatrix(prediction_gbm_tuned, data_test$Disease)
```


Comparison of all the models:

```{r}

master_table= data.table(
  Models=c("DT","ANN2","GBM","ANN1","RF","LR","ANN0","SVMNL","SVML","KNN","NBC"),
  Accuracy=c(0.7364,0.7359,0.7352,0.7334,0.731,0.7303,0.7298,0.7281,0.7272, 0.715,0.7085),
  Recall=c(0.7779,0.687,0.7644,0.7025,0.7729,0.7777,0.6869,0.7715,0.8027,0.7420,0.8451),
  Precision=c(0.7147,0.766,0.7185,0.7559,0.7095,0.7068,0.7558,0.7068,0.6939,0.7007,0.6611),
  Approx.Time=c(16.85,62.678,20.70,32.67,297,6.65,22.54,1190.5,1172.4,733.2,291.6)
  
  
)
master_table
```

In the above table, the models are arranged in the descending order with respect to the accuracy field.Based on the overall performance of the models,Decision Tree and ANN2 has the highest accuracy of 73.6% in predicting the disease for a particular patient. But while comparing the time complexity,Logistic Regresssion took the least time of 6.65 secs with the accuracy of 73%.Though the Decision Tree and ANN2 has highest accuracy with 73.6%, it took much time compared to the Logistic Regression. There is a trade off between the Accuracy and Time complexity.

##### **Testing Data Exploration**

```{r}
sapply(cardio_test, function(x) sum(is.na(x)))
cardio_test<-cardio_test[!duplicated(cardio_test),]
cardio_test<- cardio_test[cardio_test$Low.Blood.Pressure >= 20,]
cardio_test <-  cardio_test[cardio_test$Low.Blood.Pressure <= 190,]
cardio_test <-  cardio_test[cardio_test$High.Blood.Pressure>= 60,] 
cardio_test <-  cardio_test[cardio_test$High.Blood.Pressure <= 240,]
cardio_test <-  cardio_test[cardio_test$Height >= 100,]
cardio_test <-  cardio_test[cardio_test$Weight >= 20,]
```

```{r}
cardio_test$Weight<-as.numeric(cardio_test$Weight)
cardio_test$Age<-as.numeric(cardio_test$Age)
cardio_test$Height<-as.numeric(cardio_test$Height)
cardio_test$Low.Blood.Pressure<-as.numeric(cardio_test$Low.Blood.Pressure)
cardio_test$High.Blood.Pressure<-as.numeric(cardio_test$High.Blood.Pressure)
cardio_test$Smoke<-as.numeric(cardio_test$Smoke)
cardio_test$Alcohol<-as.numeric(cardio_test$Alcohol)
cardio_test$Exercise<-as.numeric(cardio_test$Exercise)

```

```{r}
str(cardio_test)
```

For ANN Model:
```{r}
rec_obj1<- recipe(ID ~ ., data = cardio_test) %>%
  step_dummy(all_nominal(), -all_outcomes()) %>%
   step_center(all_predictors(), -all_outcomes()) %>% 
  step_scale(all_predictors(), -all_outcomes()) %>% 
  prep(data = cardio_test)
rec_obj1
```


```{r}
y_test_tbl <- as.matrix(bake(rec_obj1, new_data = cardio_test) %>% select(-ID))

```


```{r}

LR <- predict(model_glm, newdata = cardio_test)
ANN0 <- model_keras0 %>% predict_classes(y_test_tbl)
ANN1<-model_keras1 %>% predict_classes(y_test_tbl)
ANN2<-model_keras2 %>% predict_classes(y_test_tbl)
DT<-predict(model_dt,newdata=cardio_test)


```

Now lets create a data frame in which we can store the prediction of the each model and also include an ID column.

```{r,eval=TRUE,results='hide',message=FALSE, warning=FALSE}

finalResults<-data.frame(LR=LR,ANN0=ANN0,ANN1=ANN1,ANN2=ANN2,DT=DT)
finalResults["ID"]<-seq.int(nrow(finalResults))
```

```{r}
write.csv(finalResults, file="C:/Users/ragha/Desktop/Priya/finalResults.csv",
          row.names=FALSE)
```
