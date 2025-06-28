SQUAREFACEAPP setup (MacOS silicon chip)

NOTE: Python3.9, Conda and created python virtualenv required!!!

-- Non-packaged run

1. Switch to your virtual environment
2. Use requirements.txt to install all modules via "pip install -r requirements.txt" (Installation into virtualenv is recommended)
3. Run via "python3 SquareFace.py"

-- Packaged run

NOTE: Valid version of PyInstaller required!!!
1. Switch to your virtual environment
2. Use requirements.txt to install all modules via "pip install -r requirements.txt"
3. To package app run following command "sudo sh pack.sh" 
4. You will need to copy "assets" directory to the following path after packaging "/YOUR-ROOT-PATH/SquareFaceApp/dist/SquareFace.app/Contents/MacOs"
5. Run SquareFace.app