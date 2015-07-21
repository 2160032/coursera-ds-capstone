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
library(slam)
source("textPrediction.R")


## constants: determined via lots of testing, evaluation, trial and error

## constant to reduce the corpus to 1/4 its initial size by random exclusion
##   too large, and the corpus won't fit in system memory
##   too small, and the variety of text is insufficient to build good ngrams
TSIZE <- 1000000 # reduce the in-memory size of vcorpus from ~16GB to 4GB

## constant representing the size of corpus used to generate ngrams
##   too large and 'tm' can't handle all the word combination permutations
##   too small and common ngrams don't occur at sufficient frequency
##    > 500000 - causes illegal memory access errors
FSIZE <- 200000 # much larger and sparse matrix creation takes too long

## constant representing the size of corpus used for accuracy testing
TSIZE<- 100


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
## copies both with and without punctuation are useful
## profanity, via http://fffff.at/googles-official-list-of-bad-words/
profanity <- as.character(read.csv("profanity.txt", header=FALSE)$V1)
filtered <- createCleanCorpus(texts.reduced, remove.punct=FALSE, remove.profanity=TRUE, profanity)
filtered.np <- createCleanCorpus(texts.reduced, remove.punct=TRUE, remove.profanity=TRUE, profanity)

# save to disk for future use
save(filtered, file="filtered-1000000-wp.Rda")
save(filtered.np, file="filtered-1000000-np.Rda")

## further reduce the size of the filtered dataset for ngram processing
filtered.sub <- sample(filtered, FSIZE, replace=FALSE)
filtered.sub.np <- sample(filtered.np, FSIZE, replace=FALSE)
filtered.test <- sample(filtered.np, TSIZE, replace=FALSE)

# remove the original sets to save ~8GB
rm(filtered)
rm(filtered.np)

# n-gram tokenizers
BigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min=2, max=2))
TrigramTokenizer <- function(x, n) NGramTokenizer(x, Weka_control(min=3, max=3))
QuadgramTokenizer <- function(x, n) NGramTokenizer(x, Weka_control(min=4, max=4))
PentagramTokenizer <- function(x, n) NGramTokenizer(x, Weka_control(min=5, max=5))

## create all the document term matrices, frequency vectors and
## word-frequency data frames. 'ft' specifies the frequency threshold for
## inclusion/optimization
options(mc.cores=1) # limit cores to prevent rweka processing problems

ft.1 <- 10
dtm.1 <- DocumentTermMatrix(filtered.sub.np, control=list(bounds=list(global=c(ft.1, Inf))))
freq.1 <- sort(col_sums(dtm.1, na.rm=T), decreasing=TRUE)
nf.1 <- data.frame(word=names(freq.1), freq=freq.1)

ft.2 <- 2
dtm.2 <- DocumentTermMatrix(filtered.sub, control=list(tokenize=BigramTokenizer, bounds=list(global=c(ft.2, Inf))))
freq.2 <- sort(col_sums(dtm.2, na.rm=T), decreasing=TRUE)
nf.2 <- data.frame(word=names(freq.2), freq=freq.2)

ft.3 <- 2
dtm.3 <- DocumentTermMatrix(filtered.sub, control=list(tokenize=TrigramTokenizer, bounds=list(global=c(ft.3, Inf))))
freq.3 <- sort(col_sums(dtm.3, na.rm=T), decreasing=TRUE)
nf.3 <- data.frame(word=names(freq.3), freq=freq.3)

ft.4 <- 2 
dtm.4 <- DocumentTermMatrix(filtered.sub, control=list(tokenize=QuadgramTokenizer, bounds=list(global=c(ft.4, Inf))))
freq.4 <- sort(col_sums(dtm.4, na.rm=T), decreasing=TRUE)
nf.4 <- data.frame(word=names(freq.4), freq=freq.4)

ft.5 <- 2
dtm.5 <- DocumentTermMatrix(filtered.sub, control=list(tokenize=PentagramTokenizer, bounds=list(global=c(ft.5, Inf))))
freq.5 <- sort(col_sums(dtm.5, na.rm=T), decreasing=TRUE)
nf.5 <- data.frame(word=names(freq.5), freq=freq.5)

# package all the ngram-frequency frames into a single object
r <- 10 # frequency span for last-resort randomization
nf <- list("f1"=nf.1, "f2"=nf.2, "f3"=nf.3, "f4"=nf.4, "f5"=nf.5, "r"=r)

## save the ngram frequencies to disk
nfname <- paste("data/nFreq", sprintf("%d", FSIZE), ft.1, ft.2, ft.3, ft.4, ft.5, sep="-")
save(nf, file=paste(nfname, "Rda", sep="."))


## product of rows*columns exceeds max vector size
##freq.2 <- sort(colSums(as.matrix(ph.dtm.2)), decreasing=TRUE)
## use slam::col_sums instead
