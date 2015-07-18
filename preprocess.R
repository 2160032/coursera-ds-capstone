## text-predictor, R/Shiny App - text corpus preprocessor
##
## (c) Patrick Charles; see http://pchuck.net
## redistribute or modify under the GPL; see http://www.gnu.org/licenses
##
## developed for the Coursera/Johns Hopkins Data Science specialization
## capstone project in collaboration with SwiftKey; see http://swiftkey.com/en/
##

## this standalone script performs the preprocessing necessary
## to create a compact and optimized set of ngram frequency dataframes
## used by the text predictor function and shiny application
library(RWeka)
library(tm)
source("textPrediction.R")


## constants: determined via lots of testing, evaluation, trial and error

## constant to reduce the corpus to 1/4 its initial size by random exclusion
##   too large, and the corpus won't fit in system memory
##   too small, and the variety of text is insufficient to build good ngrams
TSIZE <- 1000000 # reduce the in-memory size of vcorpus from ~16GB to 4GB

## constant representing the size of corpus used to generate ngrams
##   too large and 'tm' can't handle all the word combination permutations
##   too small and common ngrams don't occur at sufficient frequency
FSIZE <- 100000 # much larger and sparse matrix creation takes hours


## data sources

# read data from all sources
blogs <- readLines("data/final/en_US/en_US.blogs.txt", skipNul=TRUE)
twitter <- readLines("data/final/en_US/en_US.twitter.txt", skipNul=TRUE)
news <- readLines("data/final/en_US/en_US.news.txt", skipNul=TRUE)


## pre-processing

## combine the texts into a single vector
texts.reduced <- sample(c(blogs, news, twitter), TSIZE, replace=FALSE)
#texts.full <- c(blogs, news, twitter)
rm(blogs, twitter, news)

## perform all transformations on the data and produce a 'tm' corpus
filtered <- createCleanCorpus(texts.reduced)
##save(filtered, "filtered.Rda")

## further reduce the size of the filtered dataset for ngram processing
filtered.sub <- sample(filtered, FSIZE, replace=FALSE)

# n-gram tokenizers
BigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min=2, max=2))
TrigramTokenizer <- function(x, n) NGramTokenizer(x, Weka_control(min=3, max=3))
QuadgramTokenizer <- function(x, n) NGramTokenizer(x, Weka_control(min=4, max=4))
PentagramTokenizer <- function(x, n) NGramTokenizer(x, Weka_control(min=5, max=5))

## create all the document term matrices, frequency vectors and
## word-frequency data frames. 'ft' specifies the frequency threshold for
## inclusion/optimization
ft.1 <- 10
dtm.1 <- DocumentTermMatrix(filtered.sub, control=list(bounds=list(global=c(ft.1, Inf))))
freq.1 <- sort(colSums(as.matrix(dtm.1)), decreasing=TRUE)
nf.1 <- data.frame(word=names(freq.1), freq=freq.1)

ft.2 <- 2
dtm.2 <- DocumentTermMatrix(filtered.sub, control=list(tokenize=BigramTokenizer, bounds=list(global=c(ft.2, Inf))))
freq.2 <- sort(colSums(as.matrix(dtm.2)), decreasing=TRUE)
nf.2 <- data.frame(word=names(freq.2), freq=freq.2)

ft.3 <- 2
dtm.3 <- DocumentTermMatrix(filtered.sub, control=list(tokenize=TrigramTokenizer, bounds=list(global=c(ft.3, Inf))))
freq.3 <- sort(colSums(as.matrix(dtm.3)), decreasing=TRUE)
nf.3 <- data.frame(word=names(freq.3), freq=freq.3)

ft.4 <- 2 
dtm.4 <- DocumentTermMatrix(filtered.sub, control=list(tokenize=QuadgramTokenizer, bounds=list(global=c(ft.4, Inf))))
freq.4 <- sort(colSums(as.matrix(dtm.4)), decreasing=TRUE)
nf.4 <- data.frame(word=names(freq.4), freq=freq.4)

ft.5 <- 2
dtm.5 <- DocumentTermMatrix(filtered.sub, control=list(tokenize=PentagramTokenizer, bounds=list(global=c(ft.5, Inf))))
freq.5 <- sort(colSums(as.matrix(dtm.5)), decreasing=TRUE)
nf.5 <- data.frame(word=names(freq.5), freq=freq.5)

# package all the ngram-frequency frames into a single object
r <- 10 # frequency span for last-resort randomization
nf <- list("f1"=nf.1, "f2"=nf.2, "f3"=nf.3, "f4"=nf.4, "f5"=nf.5, "r"=r)

# save the ngram frequencies to disk
save(nf, file="nFreq.Rda") 
