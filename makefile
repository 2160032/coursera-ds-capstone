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

# remove generated files
clean:
	rm -f *.html *.md
	rm -rf *_figure/
	rm -rf *_cache/


## differences between master and gh-pages branches:

# master - .gitignore contains *.html
# gh-pages - includes: ml_project.html, .nojekyll
