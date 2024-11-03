quickbuild:
	manim-slides render presentation.py Presentation -q l
build:
	manim-slides render presentation.py Presentation -q h
convert:
	manim-slides convert Presentation out/presentation.html
clean:
	rm -rf out/*
	rm -rf media/*
	rm -rf slides/*
view:
	vivaldi out/presentation.html
run: build convert view
preview: quickbuild convert view
