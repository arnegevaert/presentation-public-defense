build:
	manim-slides render presentation.py Presentation
	manim-slides convert --to html Presentation out/presentation.html
clean:
	rm -rf out/*
	rm -rf media/*
view:
	vivaldi out/presentation.html
run: build view
