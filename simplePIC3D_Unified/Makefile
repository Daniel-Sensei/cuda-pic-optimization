.PHONY: clean
.PHONY: md5sum
.PHONY: wipe_output_data

CXX = nvcc 
EXE = simplePIC3D 
SRC = simplePIC3D.cu
INC = *.hpp

default: $(EXE)

$(EXE): $(SRC) $(INC)
	$(CXX) -o $@ $(SRC) -I ./include -L /usr/lib/x86_64-linux-gnu -L ./lib -lconfig -lfftw3 -lcurand -lcufft -arch=sm_60
	
run: $(EXE) 
	LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./lib ./$^ $(ARGS)

md5sum:
	md5sum out/global_quantities.txt

dat2xfm:
	cd ./out_h5 && sh dat2xdmf.sh && cd -

clean:
	rm -rf *.o $(EXE) 

wipe_output_data:
	rm -rf out/* out_h5/fields*.h5 out_h5/fields*.xmf
