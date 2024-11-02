scenes = Chapter1_1 Chapter1_2 Chapter1_3
quickbuild:
	manim-slides render presentation.py $(scenes) -q l
build:
	manim-slides render presentation.py $(scenes) -q k
convert:
	manim-slides convert $(scenes) out/presentation.html
clean:
	rm -rf out/*
	rm -rf media/*
	rm -rf slides/*
view:
	vivaldi out/presentation.html
run: build convert view
preview: quickbuild convert view
