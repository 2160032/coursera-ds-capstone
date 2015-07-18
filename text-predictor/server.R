## text-predictor, R/Shiny App - server
##
## (c) Patrick Charles; see http://pchuck.net
## redistribute or modify under the GPL; see http://www.gnu.org/licenses
##
## developed for the Coursera/Johns Hopkins Data Science specialization
## capstone project in collaboration with SwiftKey; see http://swiftkey.com/en/
##
library(shiny)
library(tm)
library(RWeka)

shinyServer(
    function(input, output, session) {
        source("textPrediction.R") # load the prediction functions
        #        load("nFreq.Rda") # load the ngram sparse matrices
        load("nFreq-50000-2-2-2-2-2.Rda") # load the ngram sparse matrices

        # react to text input or prediction parameter events with a prediction
        observe({
            cat(paste("predicting next for ", input$text_in, "..\n", sep=""))
            text.in <- as.character(input$text_in)
            count <- input$suggestions
            output$prediction=renderPrint(
                cat(cleanPredictNext(text.in, nf, count), sep="\n"))
        })
    }
)

