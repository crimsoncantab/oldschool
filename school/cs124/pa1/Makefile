CC=g++ -Wall -static
O=randmst
INC=-I/home/l/i/libcs124/goodies/include -L/home/l/i/libcs124/goodies/lib
LIB=-lgsl -lgslcblas

kruskals:
	$(CC) kruskals.cpp -o $(O) $(INC) $(LIB) 
	
prims:
	$(CC) $(INC) prims.cpp -o $(O) $(LIB)

clean:
	rm -f randmst
