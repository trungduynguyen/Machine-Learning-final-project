
library(caret)
library(class)
data("iris")
trainIndex <- createDataPartition(iris$Species, p = .5,list = FALSE,times = 1)
irisTrain <- iris[ trainIndex,]
irisTest  <- iris[-trainIndex,]
# train <- rbind(iris[1:25,1:4], iris[51:75,1:4], iris[101:125,1:4])
# test <- rbind(iris[26:50,1:4], iris[76:100,1:4], iris[126:150,1:4])
cl <- factor(c(rep("s",25), rep("c",25), rep("v",25)))

acc <- 1:150
for (i in 1:150){  
  # res<-knn(train, test, cl, k = i, prob=FALSE)
  res <- knn(irisTrain[1:4],irisTest[1:4],cl,k=i,prob=FALSE)
  acc[i] <- sum(as.numeric(res==cl))/75
}
#attributes(.Last.value)

plot(acc[1:50],type = 'l')

iris$Colour = "red"
iris$Colour[iris$Species=="setosa"] = "blue"
iris$Colour[iris$Species=="versicolor"] = "black"
plot(iris$Petal.Width,iris$Petal.Length, col=iris$Colour)


library(e1071)
data(iris)
m <- naiveBayes(Species ~ ., data = iris)
## alternatively:
m <- naiveBayes(iris[,-5], iris[,5])
m
table(predict(m, iris), iris[,5])



library(randomForest)
# load data
data(iris)
# fit model
fit <- randomForest(Species~., data=train,formula = )
# summarize the fit
summary(fit)
# make predictions
predictions <- predict(fit, test)
classes <- factor(c(rep("setosa",25), rep("versicolor",25), rep("virginica",25)))
# summarize accuracy
table(predictions, classes)



############################################
############################################
############################################
rm(list= ls())

setwd("C:/Users/TrungDuy/Dropbox/MLproject")
library(class)
library(corrplot)  # graphical display of the correlation matrix
library(caret)     # classification and regression training
library(klaR)      # naive bayes
library(randomForest)  # random forest, also for recursive feature elimination
library(gridExtra) # save dataframes as images
library(pROC)
library(reshape2)
library(ggplot2)
today <- as.character(Sys.Date())

wine <- read.csv('winequality-red.csv',sep = ',')


head(wine[1:11],5)

#Change the problem from regression to classification
good <- wine$quality >= 6
bad <- wine$quality < 6
wine[good, 'quality'] <- 'good'
wine[bad, 'quality'] <- 'bad'  
wine$quality <- as.factor(wine$quality) # redefine the factor variable

dummies <- dummyVars(quality ~ ., data = wine)
wine_dummied <- data.frame(predict(dummies, newdata = wine))
wine_dummied[, 'quality'] <- wine$quality

#Data splitting
set.seed(1234) 
trainIndex <- createDataPartition(wine$quality, p = .7,list = FALSE,times = 1)
wineTrain <- wine_dummied[ trainIndex,]
wineTest  <- wine_dummied[-trainIndex,]

##############################################

#Corelation matrix

numericCol <- !colnames(wineTrain) %in% c('quality')
correlMatrix <- cor(wineTrain[, numericCol])
highlyCorrelated <- findCorrelation(correlMatrix, cutoff = 0.6) # features are highly correlated if threshold > 0.6 
colnames(correlMatrix)[highlyCorrelated]

png(paste0(today, '-', 'correlation-matrix of 11 features.png'))
corrplot(correlMatrix, method = 'number', tl.cex = 0.5)
dev.off()

#wineTrain$total.sulfur.dioxide <- NULL
#wineTest$total.sulfur.dioxide <-NULL
#numericCol[7] <- FALSE

wineTrain <- wineTrain[ , -which(names(wineTrain) %in% c("total.sulfur.dioxide"))]
wineTest <- wineTest[ , -which(names(wineTest) %in% c("total.sulfur.dioxide"))]


#Normalize data to [0,1]
train_normalized <- preProcess(wineTrain[, 1:10], method = 'range')
train_plot <- predict(train_normalized, wineTrain[, 1:10])
png(paste0(today, '-', 'feature-plot.png'))
featurePlot(train_plot, wineTrain$quality, 'box')
dev.off()

#Resampling dataset
fitControl <- trainControl(method = 'cv', number = 10)

##############################################
###############NAIVEBAYES#####################
##############################################
fit_nb <- train(x = wineTrain[, 1:10], y = wineTrain$quality,
                method ='nb',trControl = fitControl)
predict_nb <- predict(fit_nb, newdata = wineTest[, 1:10])
confMat_nb <- confusionMatrix(predict_nb, wineTest$quality, positive = 'good')
importance_nb <- varImp(fit_nb, scale = TRUE)

confMat_nb

png(paste0(today, '-', 'importance-nb.png'))
plot(importance_nb, main = 'Feature importance for Naive Bayes')
dev.off()

##############################################
#####################KNN######################
##############################################
fit_knn <- train(x = wineTrain[, 1:10], y = wineTrain$quality,
                 method = 'knn',
                 preProcess = 'range', 
                 trControl = fitControl, 
                 tuneGrid = expand.grid(.k = 
                          c(3, 5, 7, 9, 11, 15, 21, 25, 31, 41, 51, 75, 101)))  
predict_knn <- predict(fit_knn, newdata = wineTest[, 1:10])
confMat_knn <- confusionMatrix(predict_knn, wineTest$quality, positive = 'good')
confMat_knn

#acc <- 1 - ((confMat_knn$table[2] + confMat_knn$table[3])) / sum(confMat_knn$table[1:4])

importance_knn <- varImp(fit_knn, scale = TRUE)

png(paste0(today, '-', 'importance-knn.png'))
plot(importance_knn, main = 'Feature importance for K-nearest neighbor')
dev.off()
##############################################
#############RANDOMFORESTS####################
##############################################
ntree <- c(1, 30, 50, 80, 120, 150, 200, 300, 500, 550, 700)
acc <- 1:11
for(i in 1:11){
  fit_rf <- train(x = wineTrain[, 1:10], y = wineTrain$quality,
                method = 'rf',
                trControl = fitControl,
                tuneGrid = expand.grid(.mtry = c(2:6)),
                ntree = ntree[i]) 
  print(fit_rf$finalModel)
  predict_rf <- predict(fit_rf, newdata = wineTest[, 1:10])
  confMat_rf <- confusionMatrix(predict_rf, wineTest$quality, positive = 'good')
  
  acc[i] <- 1 - ((confMat_rf$table[2] + confMat_rf$table[3])) / sum(confMat_rf$table[1:4])
}
#confMat_rf

acc

importance_rf <- varImp(fit_rf, scale = TRUE)

png(paste0(today, '-', 'importance-rf.png'))
plot(importance_rf, main = 'Feature importance for Random Forests')
dev.off()
##############################################

models <- resamples(list(NB = fit_nb, KNN = fit_knn,
                         RF = fit_rf))
png(paste0(today, '-', 'models-comparison.png'))
dotplot(models)
dev.off()

results <- summary(models)
png(paste0(today, '-', 'models-accuracy.png'), width = 480, height = 180)
grid.table(results$statistics$Accuracy)
dev.off()
png(paste0(today, '-', 'models-kappa.png'), width = 480, height = 180)
grid.table(results$statistics$Kappa)
dev.off()
 


Data <- data.frame(ntree, acc)
plot(acc ~ ntree, Data, pch=20,col= "red",lwd = 4,
     xlab ='Number of Trees',ylab = 'Accuracy', 
     main = 'The dependencies between Accuracy and Number of Trees')
# fit a loess line
loess_fit <- loess(acc ~ ntree, Data)
lines(Data$ntree, predict(loess_fit), col = "blue",lwd = 5)

