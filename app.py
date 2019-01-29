# Ref: https://github.com/bhavaniravi/rasa-site-bot
from flask import Flask
from flask import render_template,jsonify,request
import requests

import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy as np
import tflearn
import tensorflow as tf
import random


import pickle
data = pickle.load( open( "models/training_data", "rb" ) )
words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']

import json
with open('data/intents1.json') as json_data:
    intents = json.load(json_data)


# In[3]:


net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy')

model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')


# In[4]:


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)

    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# bag of words
def bow(sentence, words, show_details=False):
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    return(np.array(bag))


# In[5]:


bow('I want to special food', words)


# In[6]:


# load model
model.load('./models/model.tflearn')


# In[7]:






app = Flask(__name__)
app.secret_key = '12345'

@app.route('/')
def hello_world():
    return render_template('home.html')

@app.route('/learn',methods=["POST"])
def learn():
    try:
        # data structure to hold user context
        context = {}
        #sentence='is your shop open today?'
        sentence = request.form["text"]
        retResult='';
        ERROR_THRESHOLD = 0.25
        results = model.predict([bow(sentence, words)])[0]
        results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append((classes[r[0]], r[1]))
        print(return_list)#classifier result
        show_details=False
        userID='1'
        for i in intents['intents']:
            if i['tag'] == return_list[0][0]:#pass classifier result
                if 'context_set' in i:                    
                    if show_details: print ('context:', i['context_set'])
                    context[userID] = i['context_set']                    
                if not 'context_filter' in i or (userID in context and 'context_filter' in i and i['context_filter'] == context[userID]):
                    
                    if show_details: print ('tag:', i['tag'])
                    #print(random.choice(i['responses']))
                    retResult = random.choice(i['responses'])        
        return jsonify({"status":"success","response":retResult})
    except Exception as e:
        print(e)
        return jsonify({"status":"success","response":"Sorry I am not trained to do that yet..."})

@app.route('/chat',methods=["POST"])
def chat():
    try:
        user_message = request.form["text"]
        print(user_message)
        response = requests.get("http://localhost:7070/parse",params={"q":user_message})
        response = response.json()
        entities = response.get("entities")
        topresponse = response["intent"]
        intent = topresponse.get("name")
        print("Intent {}, Entities {}".format(intent,entities))
        if intent == "gst-info":
            response_text = gst_info(entities)# "Sorry will get answer soon" #get_event(entities["day"],entities["time"],entities["place"])
        elif intent == "gst-query":
            response_text = gst_query(entities)
        else:
            response_text = get_random_response(intent)
        return jsonify({"status":"success","response":response_text})
    except Exception as e:
        print(e)
        return jsonify({"status":"success","response":"Sorry I am not trained to do that yet..."})


app.config["DEBUG"] = True
if __name__ == "__main__":
    app.run(port=7070)
