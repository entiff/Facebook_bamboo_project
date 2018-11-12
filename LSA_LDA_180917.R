#=========================================================#
#### 1) Load Package ####
#=========================================================#

packages = c("Rfacebook", "tm", "lsa", "wordcloud","ggplot2","KoNLP",
             "GPArotation","cluster","RWeka","ROAuth","fpc","stringr","ape","devtools")

for (i in packages){
  if(!require( i , character.only = TRUE))
  {install.packages(i, dependencies = TRUE)}
}

#devtools::install_github('haven-jeon/KoSpacing')
library(KoSpacing) 
#if you interested in KoSpacing package, visit here (https://github.com/haven-jeon/KoSpacing)

spacing("제발이것좀띄어쓰기해주세요왜이렇게띄어쓰기를제대로안하는사람이많아요여러분들은아닐거라고믿어요글쓸때는띄어쓰기꼭합시다")

# if you have trouble with installing, check the version of java and reinstall

#=========================================================#
#### 2) Setting ####
#=========================================================#

setwd("D://Facebook_bamboo_project/data")

pdf.options(family="Korea1deb") #not to tear down the letters
options(java.parameters=c("-Xmx8g","-Dfile.encoding=UTF-8")) #to increse heap size of rjava
pal <- brewer.pal(9,"Set1")
options(mc.cores=1)

useSejongDic()

#=========================================================#
#### 3) Crawling Data ####
#=========================================================#

#### cf) issue ####
# each post has more than 17 characters
# remove meta data from post ex) year, time, post numbers and so on

Posts <- read.csv("Bamboo_posts_17v.csv",header = TRUE)
Posts <- data.frame(Posts$Sepped.v2)
Posts <- as.data.frame(Posts[1:1000,])

# load(file="topicmodeling_181106.RData")

#=========================================================#
#### 4) Preprocessing ####
#=========================================================#

#### ___1. remove pucntuation: punctuation raises error ####

posts_v1 <- as.character(Posts$`Posts[1:1000, ]`)
posts_v1 <- gsub("[[:punct:]]","",posts_v1)

# [:punct:] => [][!"#$%&'()*+,./:;<=>?@\^_`{|}~-]
# I recommend you should play with regular expression if you interested in NLP -> https://regexr.com/

#### ___2. remove redundant whitespace ####

posts_v1 <- gsub("\\s+"," ", posts_v1)

#### ___3. spacing ####

posts_v1 <- spacing(posts_v1) #more than 200 charaters, it doesn't working

#### ___4. extract noun ####

ExtractWord <- function(doc){
  
  doc <- as.character(doc)
  
  doc2 <- paste(SimplePos22(doc))
  
  doc3 <- str_match(doc2, "([가-힣]+)/NC")
  
  if(dim(doc3)[2] == 2){
    doc4 <- doc3[,2]
    doc4 <- doc4[!is.na(doc4)]
    return(doc4)
  }
}

# To know about the simplepos22 -> https://brunch.co.kr/@mapthecity/9 

nouns = sapply(posts_v1, words, USE.NAMES = F)
txt_noun1 <- nouns

#### ___5. remove stopwords ####

words <- read.csv("2nd_stopwords.txt",header=FALSE)
words <- as.character(words$V1)

#=========================================================#
#### 5) Word Embedding ####
#=========================================================#

corpus <- Corpus(VectorSource(nouns)) # assign vector value
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removeWords, words)

#=========================================================#
#### 6) LSA ####
#=========================================================#

#### ___1. Document-Term Matrix ####

uniTokenizer <- function(x) unlist(strsplit(as.character(x), "[[:space:]]+"))

control = list(tokenize = uniTokenizer,
               wordLengths=c(2,20),
               stopwords = c("\\n","\n","것","c"),
               weighting = function (x) {weightTfIdf(x, TRUE)}) 

#tf-idf = tf*idf
#tf = (특정 문서에 출현하는 문자열의 총 개수 / 특정 문서에 출현하는 문자열 a의 등장 횟수)
#idf = log(문서 총 수/문자열 a가 출현하는 문서의 개수)
#if certain word exist in many docs, reduce the count value of that word

dtm <- DocumentTermMatrix(corpus, control=control) #invert to dtm using tokenizer
object.size(dtm)

Encoding(dtm$dimnames$Terms) = "UTF-8" # encoding...ha...

findFreqTerms(dtm,lowfreq = 5)

#### ___2. Decrease Sparsity ####

td <- removeSparseTerms(dtm,0.999) # decrease matirx's sparsity using frequency
as.numeric(object.size(dt)/object.size(dtm)) * 100 # ratio of reduced/ original

td$dimnames$Terms[1:10]

#### ___3. Visualize Words (Word Cloud) ####

check <- which(rowSums(as.matrix(td))>=1) # Once we make reduced matrix, maybe there is empty row. so we need to get rid of them
td<-(td[check,])
colTotal<-apply(td,1,sum)
which(colTotal<=0) # there is no empty row, you need to identify '0'
findFreqTerms(td, lowfreq=5)

TermFreq<-colSums(as.matrix(td))
TermFreq2 <-subset(TermFreq,TermFreq>4)
gframe<-data.frame(term=names(TermFreq2),freq=TermFreq2)
ggplot(data=gframe)+aes(x=term,y=freq)+geom_bar(stat="identity")+coord_flip()
wordcloud(names(TermFreq2),TermFreq2,max.words=100,random.color=TRUE,colors=pal) # WOW ~

#### ___4. Model(LSA) ####
# review about PCA https://wikidocs.net/7646

# There is no golden rule for LSA, it depends on dataset and concern about analysis possibility

LSA <-lsa(td,dim=5) # dimensions=5
st <-LSA$tk #document ~ dimensions
wd <- LSA$dk #words ~ dimensions
strength <- LSA$sk #explanation regarding dimensions

rot <- GPForth(wd, Tmat=diag(ncol(wd)), normalize=FALSE, eps=1e-5,
               maxit=10000, method="varimax",methodArgs=NULL) #After lsa, we get 3 matrixes, and we use 'dk' matrix for varimax rotation

cord <- st %*% diag(strength) %*% rot$Th # after rotation, merge

signs <- sign(colSums(rot$loadings)) # we can get how much each word explain each dimension

cord<-cord %*% diag(signs)

text_lsa<-data.frame(cord=cord,file=file1[check])

a<-LSA$dk
which.max(a[,1])
max(a[,1])
order(a[,1])
round(abs(a[order(abs(a[,1])),1]),3)
head(abs(a[order(abs(a[,1]),decreasing = TRUE),5]))

#### ___5. Visualize Result(LSA) ####

showmedim <- function(dimen){
  t<-rot$loadings[,dimen]
  tt<-abs(t)
  terms<-names(tt)
  wordcloud(terms,tt,scale=c(4,1),rot.per=0,max.words=50, colors=pal)
}

showmedim(1)
showmedim(2)
showmedim(3)
showmedim(4)
showmedim(5)

#=========================================================#
#### 7) LDA ####
#=========================================================#


#### ___1. Model(LDA) ####

# https://ratsgo.github.io/from%20frequency%20to%20semantics/2017/06/01/LDA/

# install.packages("lda")
# install.packages("topicmodels")

library(lda)
library(topicmodels)

# There is no perplexity method for collased gibbs sampling
# So, let's find proper k in Gibbs method

result.lda_20 <- LDA(td, k = 20, method = "Gibbs",
                   control = list(burnin = 1000, iter = 3000, keep = 50))

result.lda_30 <- LDA(td, k = 30, method = "Gibbs",
                   control = list(burnin = 1000, iter = 3000, keep = 50))

result.lda_40 <- LDA(td, k = 40, method = "Gibbs",
                     control = list(burnin = 1000, iter = 3000, keep = 50))

result.lda_50 <- LDA(td, k = 50, method = "Gibbs",
                   control = list(burnin = 1000, iter = 3000, keep = 50))

perplexity(result.lda_20, newdata = td) # 503
perplexity(result.lda_30, newdata = td) # 448
perplexity(result.lda_40, newdata = td) # 404
perplexity(result.lda_50, newdata = td) # 374

# we don't have such computing power like you..
# so we just test k four times(k = 20, 30, 40, 50)
# based on perplexity, k = 50

ldaform <- dtm2ldaformat(td, omit_empty=TRUE)

# alpha, eta are hyperparameters but use 0.01 in general

result.lda <- lda.collapsed.gibbs.sampler(documents = ldaform$documents,
                                          K = 50, vocab = ldaform$vocab,
                                          num.iterations = 5000, burnin = 1000,
                                          alpha = 0.01, eta = 0.01)

#=====================================================================#
# Cf. parameters & options                                            #
#                                                                     #
#     1. num.iterations => posteria number of update                  #
#     2. burnin = burning of first value                              #
#     3. alpha = probability of topics in document / 1 = uniform      #
#     4. eta = parameter setting = probability of words in one topic  #
#                                                                     #
#=====================================================================#

attributes(result.lda)
dim(result.lda$topics) 
result.lda$topics
top.topic.words(result.lda$topics) # 20 words per topic
result.lda$topic_sums #how many words 

#### ___2. Visualization ####

alpha = 0.01
eta = 0.01

theta <- t(apply(result.lda$topic_sums + alpha, 2, function(x) x/sum(x)))
phi <- t(apply(t(result.lda$topics) + eta, 2, function(x) x/sum(x)))

Bamboo_topics <- list(phi = phi,
                     theta = theta,
                     doc.length = 83857,
                     vocab = ldaform$vocab,
                     term.frequency = TermFreq)

options(encoding = 'UTF-8') # encoding ... ha...

#install.packages("LDAvis")
library(LDAvis)

# create the JSON object to feed the visualization
# defalut method = PCA

json <- createJSON(phi = Bamboo_topics$phi,
                   theta = Bamboo_topics$theta, 
                   doc.length = Bamboo_topics$doc.length, 
                   vocab = Bamboo_topics$vocab, 
                   term.frequency = Bamboo_topics$term.frequency, encoding='UTF-8')

#install.packages("servr")
library(servr)

serVis(json, out.dir = 'vis', open.browser = TRUE)

#==========================================================#
#################### End!!! DO SAVE !!! ####################
#==========================================================#

# save.image(file = "topicmodeling_181106.RData")

#==========================================================#