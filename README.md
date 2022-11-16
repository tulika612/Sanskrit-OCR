# Parinamika
State of the art OCR for Sanskrit Text

## Installation
1. Clone the repository <br/>
2. Download the model files from Google Drive: https://drive.google.com/file/d/1KJ6vORY-Ybi_ldvdj2cDGAYsnRG4wdCR/view <br/>
3. Extract them and place them in the modelss folder <br/> 
4. Create a new virtual environment and activate it. <br/>
5. After that, install all the packages required using the following command in the CLI:
```
pip install -r requirements.txt
```
This should install the required packages.

## Running the application on localhost:
To run the application, enter the CLI and type the following commands:
```
python manage.py runserver
```
Navigate to http://127.0.0.1:8000/ in any browser, the application will be running.

## Creating Pull Requests
Before creating Pull Requests, don't forget to add the model files (shown below) to the .gitignore file:
```
mysite/modelss/model.ckpt-310100.data-00000-of-00001
mysite/modelss/model.ckpt-310100.index
mysite/modelss/model.ckpt-310100.meta
```
