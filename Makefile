CA:
	g++ CAreservoir.cpp -O3  -fopenmp -c -o CAreservoir.o
	g++ CAreservoir.o -O3  -fopenmp alglib_func.a -o CAreservoir


clean:
	rm -f CAreservoir.o *.dat SVM_model* SVM_results* *.csv
