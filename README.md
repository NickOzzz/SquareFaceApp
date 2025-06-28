SQUAREFACEAPP setup (MacOS silicon chip)


-- Local run without package

NOTE: Python3.9 and Conda required!!!
1. Use requirements.txt to install all modules via "pip install -r requirements.txt" (Installation into virtualenv is recommended)
2. Run via "python3 SquareFace.py"

-- Packaged run

NOTE: Created virtualenv and valid version of PyInstaller required!!!
1. To package app run following command "sudo sh pack.sh /PATH/TO/YOUR/VIRTUALENV/SITEPACKAGES" 
2. You will need to copy "assets" directory to the following path after packaging "/YOUR-ROOT-PATH/SquareFaceApp/dist/SquareFace/Contents/MacOs"
3. Run SquareFace.app