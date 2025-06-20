all: clean run analysis

run: 
	python src/main.py

test: clean run 

clean:
	del /Q data\clean\* 
	del /Q data\plots\*
	del /Q data\year\rookie\* 
	del /Q data\year\* 

analysis:
	python src/analysis.py
