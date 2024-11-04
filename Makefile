quickbuild:
	manim-slides render presentation.py Presentation -q l
	manim-slides convert Presentation out/presentation.html
build:
	manim-slides render presentation.py Presentation -q h
	manim-slides convert Presentation out/presentation.html
clean:
	rm -rf out/*
	rm -rf media/*
	rm -rf slides/*
view:
	vivaldi out/presentation.html
run: build view
preview: quickbuild view
