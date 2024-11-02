build:
	manim-slides convert --to html Presentation out/presentation.html
clean:
	rm -rf out/*
	rm -rf media/*
