# Coursera, Data Science Capstone
#
# natural language processing
#

# tmuxinator an R dev environment
create_env:
	tmuxinator start r-sandbox

MILESTONE=milestone
milestone:
	Rscript -e "rmarkdown::render('$(MILESTONE).Rmd', 'html_document', '$(MILESTONE).html'); browseURL('$(MILESTONE).html')"

FINAL=capstone
final:
	Rscript -e "rmarkdown::render('$(FINAL).Rmd', 'html_document', '$(FINAL).html'); browseURL('$(FINAL).html')"


# fetch profanity list
# (minor formatting changes and saved to data/profanity.txt)
get_profanity:
	wget https://gist.github.com/jamiew/1112488

# run shiny server locally
run_app:
#	cp nFreq.Rda text-predictor/
	R -e "shiny::runApp('text-predictor', display.mode='showcase')"

# deploy to shinyapps.io
deploy_app:
	R -e "shinyapps::deployApp('text-predictor')"

# slides (apply on branch gh-pages)
slidify:
	R -e "slidify::slidify('index.Rmd')"

# view slides locally
view_slides:
	R -e "browseURL('index.html')"

# remove generated files
clean:
	rm -f *.html *.md
	rm -rf *_figure/
	rm -rf *_cache/


## differences between master and gh-pages branches:

# master - .gitignore contains *.html
# gh-pages - includes: ml_project.html, .nojekyll
