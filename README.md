# An-Arabic-Chatbot-with-speech-recognition-Feature-Using-Tensorflow
An Arabic chatbot built using multiple libraries such as Tensorflow, tflearn, NLTK, and others. It recognizes speech then translates it to text(stt), and processes text to speech (tts).

## The Projects Component:
1. A JSON file that holds the data for the chatbot.
2. A Python code that loads the data, applies NLP on it, trains a ML model, and starts chatting with the user.
3. The chatting process can be done using the keyboard or the stt and tts features.

## The NLP process:
1. Tokenization (splitting the sentence)
2. Remove stop_words (الى, من)
3. Normalization using camel tools (أ --> ا)
4. Stemmeraization (root word)

## Download process:
1. Install Anaconda (Spyder) platform.
2. Open Anaconda Terminal.
3. Create a virtual environment that supports python 3.6:
```
conda create -n py36 python=3.6
activate py36
```
4. Install the needed modules in the environment:
```
pip install nltk
pip install numpy
pip install tflearn
pip install tensorflow
pip install camel_tools
pip install gtts
pip install speechRecognition
pip install eyed3
pip install spyder
pip install pyaudio
```
5. Type "spyder" to launch the application.
6. open the python script included in the repository.
7. By default, the downloaded script will run as a vocal chatbot.
8. To use the keyboard add a # to the left of the lines 146, 151, 155, 168, and delete the # in line 150.
9. Run the program. 
