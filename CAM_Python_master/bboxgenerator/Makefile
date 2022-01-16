all: cut dt_box


cut:
	g++ -g -O3 gc.cpp -o cut `pkg-config --libs opencv` -lm -I /dccstor/alfassy/anaconda32/envs/kef/include -L /dccstor/alfassy/anaconda32/envs/kef/lib -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -Wl,-rpath,/dccstor/alfassy/anaconda32/envs/kef/lib

dt_box:
	g++ -g -O3 dt.c dt_box.cpp -o dt_box `pkg-config --libs opencv` -lm -I /dccstor/alfassy/anaconda32/envs/kef/include -L /dccstor/alfassy/anaconda32/envs/kef/lib -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -Wl,-rpath,/dccstor/alfassy/anaconda32/envs/kef/lib

.PHONY: clean
clean:
	rm cut dt_box
