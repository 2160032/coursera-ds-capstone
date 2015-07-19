## text-predictor, R/Shiny App - ui
##
## (c) Patrick Charles; see http://pchuck.net
## redistribute or modify under the GPL; see http://www.gnu.org/licenses
##
## developed for the Coursera/Johns Hopkins Data Science specialization
## capstone project in collaboration with SwiftKey; see http://swiftkey.com/en/
##
library(shiny)

shinyUI(navbarPage("Text Predictor",
  tabPanel("Predict",
    fluidRow(
      column(4,
        h3('Input'), 
        tags$textarea(id="text_in", rows=3, cols=30),
        h4('Prediction Parameters'), 
        sliderInput("suggestions", "Word Suggestions", 
                    value=1.0, min=1.0, max=5.0, step=1.0)
      ),
      column(2,
        h3("Current Word"),
        HTML("<br>"),
        verbatimTextOutput('word.current')
      ),
      column(2,
        h3("Next Word"),
        HTML("<br>"),
        verbatimTextOutput('word.next')
      )
    )
  ),
  tabPanel("About", HTML("")
  )
))
