
# Next word prediction [EN][VI]
Originally forked from [Git](https://github.com/renatoviolin/next_word_prediction).
### Application
Simple application using transformers models to predict next word or a masked word in a sentence. Inference time is acceptable even in CPU.

This app implements two variants of the same task (predict <mask> token). The first one considers the <mask> is at end of the sentence, simulating a prediction of the next word of the sentence.
The second variant is necessary to include a <mask> token where you want the model to predict the word.

------------------------------------------------------------
### Requirement-steps:
#### 0. Install Python:
 [![PythonVersion](https://img.shields.io/static/v1?label=python&message=3.7%20|%203.9&color=blue)](https://www.python.org/downloads/)
First things first, you'll need to install Python (3.9.4 is the newest stable version when writing this).
*Consider adding Python to PATH (in options when installing, or later - you can google it) for easier python command execution.*


 * You will need to do below steps using command-line in the application folder. 
#### 1. Create and active virtual environment:
```bash
python -m venv venv
venv\scripts\activate
```
#### 2. Install requirements:
```bash
pip install requirements.txt
```
#### 3. Run application:
*The first load takes a long time since the application will download all the models.*
```bash
python app.py
```
Open your browser at http://localhost:8000.