# Next word prediction [EN][VI]
Originally forked from [Git](https://github.com/renatoviolin/next_word_prediction).
### Application
Simple application using transformers models to predict next word or a masked word in a sentence. Inference time is acceptable even in CPU.

This app implements two variants of the same task (predict <mask> token). The first one considers the <mask> is at end of the sentence, simulating a prediction of the next word of the sentence.
The second variant is necessary to include a <mask> token where you want the model to predict the word.

------------------------------------------------------------
### Requirement-steps:
 *You will need to do this using command-line in application folder. *
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