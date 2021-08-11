# -*- coding: utf-8 -*-
import nltk
import numpy
import tflearn
import tensorflow
import random
import json
from nltk.stem.isri import ISRIStemmer
from nltk.corpus import stopwords
from camel_tools.utils.normalize import normalize_alef_ar 
from gtts import gTTS
import os
import time
import speech_recognition as sr
import eyed3
#--------------------------------DATA--------------------------------
with open("intents.json", encoding="utf-8") as file:  #load the json file content
    data = json.load(file)

words = []
labels = []
docs_x = []
docs_y = []


#--------------------------------NLP--------------------------------
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern) #tokenize the pre_written patterns in the json file
        words.extend(wrds)  #add ALL the tokenized patterns in a list
        docs_x.append(wrds)
        docs_y.append(intent["tag"]) #connect the patterns saved in docs_x with their corresponding tag

    if intent["tag"] not in labels: #add the tags into a list
        labels.append(intent["tag"])


stop_words= list(stopwords.words("Arabic")) #save the arabic stop words in a list

#normalize the words in the lists 
stop_words=[normalize_alef_ar(w) for w in stop_words] #Alef variations to plain a Alef character like (ء)
words=[normalize_alef_ar(w) for w in words] 
            
[words.remove(w) for w in words if w in stop_words] #remove the stop words from the list

st = ISRIStemmer() # Stemmer function 
words = [st.stem(w) for w in words] #apply Stemmer on the words included in the list

words = sorted(list(set(words)))

labels = sorted(labels)


#--------------------------------MODEL DATA--------------------------------
training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x): #create the bag of words (matrics)
    bag = []

    wrds = [st.stem(w) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)


training = numpy.array(training)
output = numpy.array(output)


#--------------------------------MODEL--------------------------------
tensorflow.compat.v1.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])]) #Input layer
#Two hidden layer
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax") #output layer (softmax is used to give probability)
net = tflearn.regression(net)

model = tflearn.DNN(net) #train the model


model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True) #pass the training data
model.save("model.tflearn") #save the model
   

#--------------------------------CHATBOT--------------------------------
def stt_function(): # speech_recognition (speech to text) function

    r = sr.Recognizer()
    
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source) #remove noises

        audio = r.listen(source)
        try:
            stt=r.recognize_google(audio, language="ar") # recognize the speech using google speech
        except:
            stt=""
            
        print (stt) 
        
    return stt

def tts_function(response):
    tts= gTTS(response, lang="ar")
    tts.save("response.mp3")
    duration = eyed3.load('response.mp3').info.time_secs
    os.system("response.mp3")
    time.sleep(duration)
    os.remove("response.mp3")
    
def bag_of_words(s, words): #process the user's input
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    
    s_words=[normalize_alef_ar(w) for w in s_words] 
    
    [s_words.remove(w) for w in s_words if w in stop_words]
    
    s_words = [st.stem(w) for w in s_words]
    
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
                
    return numpy.array(bag)


def chat(): #start the chating process
    print("مرحبا, انا عَون مسؤول عن استقبال ارائكم عن الحفل الختامي")
    tts_function("مرحبا, انا عَون مسؤول عن استقبال ارائكم عن الحفل الختامي")
    
    while True:
        print("انت:")
        #inp = input() #get the input from the user through the keyboard
        inp=stt_function() #This line is used instead of the above command to get the users input through speech
        
        if inp == "خروج": #the chatbot's stop condition
            print("اسعد دائما باستقبال تقييم عملائنا وضيوفنا العزيزين")
            tts_function("اسعد دائما باستقبال تقييم عملائنا وضيوفنا العزيزين")
            break
        
        results = model.predict([bag_of_words(inp, words)])
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']
        response=random.choice(responses)   
        
        print(response)
        tts_function(response)

chat()