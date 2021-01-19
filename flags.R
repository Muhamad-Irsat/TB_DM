countries <- read.csv("D:/Data Mining TB/Flag-dataset/dataset/flagswheaders.csv", stringsAsFactors=FALSE)

str(countries)
flag <- countries 
names(flag)
colnames(flag)[25] <- "triangle"

#categorical variables to binary
for(level in unique(flag$dominant)){
  flag[paste("dominant", level, sep = "_")] <- ifelse(flag$dominantcolour == level, 1, 0)
}
flag = subset(flag, select=-c(dominantcolour))

for(level in unique(flag$topleftcolour)){
  flag[paste("topleft", level, sep = "_")] <- ifelse(flag$topleftcolour == level, 1, 0)
}
flag = subset(flag, select=-c(topleftcolour))

for(level in unique(flag$botrightcolor)){
  flag[paste("botright", level, sep = "_")] <- ifelse(flag$botrightcolor == level, 1, 0)
}
flag = subset(flag, select=-c(botrightcolor))

flag = subset(flag, select=-c(country,landmass,zone,area,population,language  ))


str(flag)

l <- c(0,1,2,3,4,5,6,7)

barplot(table(factor(countries$religion, levels = l)) , col = c('Black', 'Navy', 'Brown', 'Gold', 'Green', 'Orange', 'Red', 'White',  'Purple'), xlab ='Religion', names.arg=c('Catholic', 'Other Chri', 'Muslim', 'Buddhist', 'Hindu', 'Ethnic',
                                                                                                                                                                               'Marxist', 'Others'),  cex.names=1.5, ylab = 'Number Of Flags')

l <- c(1,2,3,4,5,6)
barplot(table(factor(countries$landmass, levels = l)) , col = c('Black', 'Navy', 'Brown', 'Gold', 'Green', 'Orange', 'Red', 'White',  'Purple'), xlab ='Landmass', names.arg=c('N.America', 'S.America', 'Europe', 'Africa',
                                                                                                                                                                               'Asia', 'Oceania'),  cex.names=1.5, ylab = 'Number Of Flags')



l <- c(1,2,3,4,5,6,7,8,9,10)
barplot(table(factor(countries$language, levels = l)) , col = c('Black', 'Navy', 'Brown', 'Gold', 'Green', 'Orange', 'Red', 'White',  'Purple'), xlab = 'Language', names.arg=c('English', 'Spanish', 'French', 'German','Slavic', 'Ind-Eur', 'Chinese','Arabic', 'Ja/Tu...','Others'),  cex.names=1.5, ylab = 'Number Of Flags')

barplot(table(factor(countries$dominantcolour)) , col = c('Black', 'Navy', 'Brown', 'Gold', 'Green', 'Orange', 'Red', 'White',  'Purple'), xlab = 'Dominant Color',  cex.names=1.5, ylab = 'Number Of Flags')

barplot(table(factor(countries$colours)) , col = c('Black', 'Navy', 'Brown', 'Gold', 'Green', 'Orange', 'Red', 'White',  'Purple'), xlab = 'Number of different colours', cex.names=1.5, ylab = 'Number Of Flags')

set.seed(7)
# load the library
library(mlbench)
library(caret)
# prepare training scheme
control <- trainControl(method="repeatedcv", number=10, repeats=3)
# train the model
model <- train(as.factor(religion)~., data=flag, method="lvq", trControl=control)
# estimate variable importance
importance <- varImp(model, scale=FALSE)
# summarize importance
print(importance)
# plot importance
plot(importance)

library(Boruta)

set.seed(123)
boruta.train <- Boruta(as.factor(religion)~., data = flag, doTrace = 2)
print(boruta.train)


dev.off()
plot(boruta.train, xlab = "", xaxt = "n")
lz<-lapply(1:ncol(boruta.train$ImpHistory),function(i)
  boruta.train$ImpHistory[is.finite(boruta.train$ImpHistory[,i]),i])
names(lz) <- colnames(boruta.train$ImpHistory)
Labels <- sort(sapply(lz,median))
axis(side = 1,las=2,labels = names(Labels),
     at = 1:ncol(boruta.train$ImpHistory), cex.axis = 0.7)


final.boruta <- TentativeRoughFix(boruta.train)
print(final.boruta)

dev.off()
plot(final.boruta, xlab = "", xaxt = "n")
lz<-lapply(1:ncol(final.boruta$ImpHistory),function(i)
  final.boruta$ImpHistory[is.finite(final.boruta$ImpHistory[,i]),i])
names(lz) <- colnames(final.boruta$ImpHistory)
Labels <- sort(sapply(lz,median))
axis(side = 1,las=2,labels = names(Labels),
     at = 1:ncol(final.boruta$ImpHistory), cex.axis = 0.7)

getSelectedAttributes(final.boruta, withTentative = F)

library(randomForest)

rf <- randomForest(as.factor(religion)~.,data=flag, trials=10, method='class') 

varImpPlot(rf,  
           sort = T,
           n.var=20,
           main="Top 20 - Variable Importance")

rfact <- flag$religion
rfpred <- predict(rf, data=flag, type='class')
rfpred <- unname(rfpred)

table(rfact, rfpred)
prop.table(table(rfact, rfpred),2)

rfmisClasificError <- mean(rfact != rfpred) 
print(paste('Accuracy',1-rfmisClasificError))


highcor <- subset(flag, select = c(religion,           
                                   crosses, white, botright_green, blue, botright_blue, dominant_blue, quarters, saltires,
                                   green, circles, bars, stripes, colours, gold, black, crescent, animate, topleft_white
))


pca <- prcomp(highcor[,-1],
              center = TRUE,
              scale. = TRUE) 
print(pca)
# plot method
plot(pca, type = "l")
summary(pca)

# Predict PCs
predict(pca, 
        newdata=tail(highcor[,-1], 2))

plot(pca)

pca2 =prcomp(highcor[,-1])
summary(pca2)
plot(pca2)

plot(pca, type = "l")

pcahighcor <- data.frame(highcor$religion, pca$x)
colnames(pcahighcor)[1] <- "religion"

pcahighcor <- pcahighcor[,1:10]

library(caret)
set.seed(3456)
trainIndex <- createDataPartition(pcahighcor$religion, p = .8,
                                  list = FALSE,
                                  times = 1)
head(trainIndex)

pcatrain <- pcahighcor[ trainIndex,]
pcatest  <- pcahighcor[-trainIndex,] 

str(pcatrain)

library(rpart)
rpart.model <- rpart(as.factor(religion) ~ .,data = pcahighcor,method='class',control=rpart.control(minsplit=10))

rpartact <- pcatest$religion
rpartpred <- predict(rpart.model2, newdata=pcatest, type='class')
rpartpred <- unname(rpartpred)

table(rpartact, rpartpred)
prop.table(table(rpartact, rpartpred),2)

rpartmisClasificError <- mean(rpartact != rpartpred) 
print(paste('Accuracy',1-rpartmisClasificError))


ecoli.rpart2 = prune(ecoli.rpart1, cp = 0.02)

plot(rpart.model, uniform = TRUE)
text(rpart.model, cex = 0.75)
print(rpart.model)
printcp(rpart.model2)
summary(rpart.model)
rpart.model2 = prune(rpart.model, cp = 0.015)
plot(rpart.model2, uniform = TRUE)
text(rpart.model2, cex = 0.75)

library(ipred)
# fit model
fit <- bagging(as.factor(religion) ~., data=pcatrain, dat=rodat,nbagg=30,coob=T, method='class')
fitact <- pcatest$religion
fitpred <- predict(fit, newdata=pcatest, type='class')

table(fitact, fitpred)
prop.table(table(fitact, fitpred),2)

fitmisClasificError <- mean(fitact != fitpred) 
print(paste('Accuracy',1-fitmisClasificError))


library(C50)
# fit model
C5 <- C5.0(as.factor(religion) ~., data=pcatrain, trials=10, method='class')
C5act <- pcatest$religion
C5pred <- predict(C5, newdata=pcatest, type='class')
#C5pred <- round(unname(C5pred))

table(C5act, C5pred)
prop.table(table(C5act, C5pred),2)

C5misClasificError <- mean(C5act != C5pred) 
print(paste('Accuracy',1-C5misClasificError))
summary(C5)

library(RWeka)


fitpart <- PART(as.factor(religion) ~., data=pcatrain)
partact <- pcatest$religion
partpred <- predict(fitpart, newdata=pcatest, type='class')


table(partact, partpred)
prop.table(table(partact, partpred),2)

partmisClasificError <- mean(partact != partpred) 
print(paste('Accuracy',1-partmisClasificError))


library(RWeka)

# fit model
J48 <- J48(as.factor(religion) ~., data=pcatrain)
J48act <- pcatest$religion
J48pred <- predict(J48, newdata=pcatest, type='class')

table(J48act, J48pred)
prop.table(table(J48act, J48pred),2)

J48misClasificError <- mean(J48act != J48pred) 
print(paste('Accuracy',1-J48misClasificError))
plot(J48)

library(MASS)

# fit model
fitlda <- lda(as.factor(religion) ~., data=pcatrain, trials=10)
partact <- pcatest$religion
partpred <- predict(fitlda, newdata=pcatest)$class

table(partact, partpred)
prop.table(table(partact, partpred),2)

partmisClasificError <- mean(partact != partpred) 
print(paste('Accuracy',1-partmisClasificError))

library(randomForest)

rf <- randomForest(as.factor(religion)~.,pcatrain, trials=10, method='class') 

rfact <- pcatrain$religion
rfpred <- predict(rf, data=pcatest, type='class')
rfpred <- unname(rfpred)

table(rfact, rfpred)
prop.table(table(rfact, rfpred),2)

rfmisClasificError <- mean(rfact != rfpred) 
print(paste('Accuracy',1-rfmisClasificError))


library(e1071)
svm <- svm(as.factor(religion)~. ,data=pcatrain, trials=10, method='class') 
svmact <- pcatest$religion
svmpred <- predict(svm, newdata=pcatest,  type='class')
svmpred <- unname(svmpred)

table(svmact, svmpred)
prop.table(table(svmact, svmpred),2)

svmmisClasificError <- mean(svmact != svmpred) 
print(paste('Accuracy',1-svmmisClasificError))


library("data.table")
library(mclust)

dtFeatures <- data.table(pcahighcor[,-1])
arrayFeatures <- names(dtFeatures)
dtFeaturesKm <- dtFeatures[, arrayFeatures, with=F]
matrixFeatures <- as.matrix(dtFeaturesKm)

nCenters <- 8
modelKm <- kmeans(  x = matrixFeatures,  centers = nCenters  )
dtFeatures[, clusterKm := modelKm$cluster]

clusters <- modelKm$cluster
clusters <- as.data.frame(clusters)
colnames(clusters)[1] <- "kmeans"

hc1<-hclust(dist(matrixFeatures))
hc1<-hclust(dist(matrixFeatures), method = 'average')

plot(hc1)
rect.hclust(hc1, k=8, border="red")
summary(hc1)
clusterCut1 <- cutree(hc1, 8)

plot(silhouette(cutree(hc1, 8),dist(matrixFeatures)))


clPairs(matrixFeatures, clusterCut1)
dtFeatures[, clusterHc := clusterCut1]

clusters$hierarchical <- clusterCut1

table(clusterCut1, flag$religion+1)

prop.table(table(clusterCut1, flag$religion+1),2)

hcmisClasificError <- mean(clusterCut1 != (flag$religion+1)) 
print(paste('Accuracy',1-hcmisClasificError))


groups.8 = cutree(hc1,8)
table(groups.8)

counts = sapply(2:8,function(ncl)table(cutree(hc1,ncl)))
names(counts) = 2:8
counts


library(cluster)
set.seed(1234)
pam1 <- pam(matrixFeatures, k = 8)
pam1

plot(pam1$clustering)
clPairs(matrixFeatures, pam1$clustering)



clusters$pam <- pam1$clustering

dtFeatures[, clusterPam := pam1$clustering]

table(groups.8,pam1$clustering)

table(pam1$clustering)

counts = sapply(2:8,function(ncl)table(pam(matrixFeatures, ncl)))
names(counts) = 2:8
counts

plot(pam1)


library(mclust)

set.seed(4)
mc1<-Mclust(matrixFeatures, 8)
plot(mc1$classification)
clPairs(matrixFeatures, mc1$classification)

clusters$mclust <- mc1$classification


dtFeatures[, clusterMc := mc1$classification]

clusters$mclust <- mc1$classification

table(mc1$classification)


res2<-Mclust(matrixFeatures)
summary(res2)             


res2$G	

str(round(res2$z,2) )	 
# assign each observation to a cluster
str(round(res2$classification,0)	 )
# for each observation show the uncertainty of the cluster assignment
str(round(res2$uncertainty,2)	   )


####

dev.off()
plot(res2, what="BIC")


#### 

clPairs(matrixFeatures, res2$classification)

dtFeatures[, Mclust := res2$classification]


plot(res2, what = "classification")

dtFeatures$clusterMclust <- as.factor(res2$classification)
factor(table(dtFeatures$clusterMclust))

lapply(split(dtFeatures$clusterMclust, dtFeatures$country), unique)

str(dtFeatures)
####

nameCluster <- 'Mclust'
dtFeatures[, list(.N), by=nameCluster]

dtFeatures[, list(nCountries=.N), by=nameCluster]

names(dtFeatures)

dtFeatures$language <- countries$language
dtFeatures$country <- countries$country

dtFeatures$landmass <- countries$landmass
dtFeatures$religion <- countries$religion

library(plyr)
dtFeatures$language <- as.factor(dtFeatures$language)
dtFeatures$language<- revalue(dtFeatures$language, c( '1'='English', '2'='Spanish', '3'='French', '4'='German', '5'='Slavic', '6'='OtherIndo-European', '7'='Chinese', '8'='Arabic', 
                                                      '9'='Japanese/Turkish/Finnish/Magyar', '10'='Others'))
dtFeatures[, table(language)]
nameCluster <- 'Mclust'
dtFeatures[, as.list(table(language)), by=nameCluster]

dtFeatures[, as.list(table(language)/.N), by=nameCluster]

dtClusters <- dtFeatures[  , c(list(nCountries=.N), as.list(table(language) / .N)),  by=nameCluster  ]

# visualize 
arrayLanguages <- dtFeatures[, unique(language)]
dtBarplot <- dtClusters[, arrayLanguages, with=F]
matrixBarplot <- t(as.matrix(dtBarplot))
nBarplot <- c(1,2,3)
namesLegend <- names(dtBarplot)

namesLegend <- substring(namesLegend, 1, 12)
arrayColors <- rainbow(length(namesLegend))
plotTitle <- paste('languages in each cluster of', nameCluster)

# build the histogram
barplot(  height = matrixBarplot,  names.arg = nBarplot,  col = arrayColors,  legend.text = namesLegend,  xlim = c(0, ncol(matrixBarplot) * 2),  main = plotTitle,  xlab = 'cluster')

########
# define a function to build the histogram
plotCluster <- function(  dtFeatures,   nameCluster ){
  
  # aggregate the data by cluster  
  dtClusters <- dtFeatures[    , c(list(nCountries=.N), as.list(table(language) / .N)),    by=nameCluster]    
  # prepare the histogram inputs  
  arrayLanguages <- dtFeatures[, unique(language)]  
  dtBarplot <- dtClusters[, arrayLanguages, with=F]  
  matrixBarplot <- t(as.matrix(dtBarplot))  
  nBarplot <- dtClusters[, nCountries]  
  namesLegend <- names(dtBarplot)  
  namesLegend <- substring(namesLegend, 1, 12)  
  arrayColors <- rainbow(length(namesLegend))    
  # build the histogram  
  barplot(    height = matrixBarplot,    
              names.arg = nBarplot,    
              col = arrayColors,    
              legend.text = namesLegend,    
              xlim=c(0, ncol(matrixBarplot) * 2),    
              main = paste('languages in each cluster of', nameCluster),    
              xlab = 'cluster'  )  }

plotCluster(dtFeatures, nameCluster)



###### visualize landmass 
####################################################################
dtFeatures$landmass <- as.factor(dtFeatures$landmass)
dtFeatures$landmass<- revalue(dtFeatures$landmass, c( '1'='N.America', '2'='S.America', '3'='Europe', '4'='Africa', '5'='Asia', '6'='Oceania'))

dtFeatures[, table(landmass)]
nameCluster <- 'Mclust'
dtFeatures[, as.list(table(landmass)), by=nameCluster]

dtFeatures[, as.list(table(landmass)/.N), by=nameCluster]

dtClusters <- dtFeatures[  , c(list(nCountries=.N), as.list(table(landmass) / .N)),  by=nameCluster  ]

# visualize 
arraylandmasss <- dtFeatures[, unique(landmass)]
dtBarplot <- dtClusters[, arraylandmasss, with=F]
matrixBarplot <- t(as.matrix(dtBarplot))
nBarplot <- c(1,2,3)
namesLegend <- names(dtBarplot)

namesLegend <- substring(namesLegend, 1, 12)
arrayColors <- rainbow(length(namesLegend))
plotTitle <- paste('landmass in each cluster of', nameCluster)

# build the histogram
barplot(  height = matrixBarplot,  names.arg = nBarplot,  col = arrayColors,  legend.text = namesLegend,  xlim = c(0, ncol(matrixBarplot) * 2),  main = plotTitle,  xlab = 'cluster')


###### visualize religion
####################################################################
dtFeatures$religion <- as.factor(dtFeatures$religion)
dtFeatures$religion<- revalue(dtFeatures$religion, c( '0'='Catholic', '1'='Other Christian', '2'='Muslim', '3'='Buddhist', '4'='Hindu',
                                                      '5'='Ethnic', '6'='Marxist', '7'='Others'))

dtFeatures[, table(religion)]
nameCluster <- 'Mclust'
dtFeatures[, as.list(table(religion)), by=nameCluster]

dtFeatures[, as.list(table(religion)/.N), by=nameCluster]

dtClusters <- dtFeatures[  , c(list(nCountries=.N), as.list(table(religion) / .N)),  by=nameCluster  ]

# visualize 
arrayreligions <- dtFeatures[, unique(religion)]
dtBarplot <- dtClusters[, arrayreligions, with=F]
matrixBarplot <- t(as.matrix(dtBarplot))
nBarplot <- c(1,2,3)
namesLegend <- names(dtBarplot)

namesLegend <- substring(namesLegend, 1, 12)
arrayColors <- rainbow(length(namesLegend))
plotTitle <- paste('religion in each cluster of', nameCluster)

# build the histogram
barplot(  height = matrixBarplot,  names.arg = nBarplot,  col = arrayColors,  legend.text = namesLegend,  xlim = c(0, ncol(matrixBarplot) * 2),  main = plotTitle,  xlab = 'cluster')




