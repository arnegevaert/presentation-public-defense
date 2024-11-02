quickbuild:
	manim-slides render presentation.py Presentation -q l
	manim-slides convert --to html Presentation out/presentation.html
build:
	manim-slides render presentation.py Presentation -q k
	manim-slides convert --to html Presentation out/presentation.html
clean:
	rm -rf out/*
	rm -rf media/*
view:
	vivaldi out/presentation.html
run: build view
preview: quickbuild view
