import nltk
import pickle
import json
import numpy as np
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation
import random
import tensorflow
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()

words=[]
classes=[]
documents=[]
ignore=['?','!',',',"'s","."]

data_file=open("C:\\Users\\chinm\\OneDrive\\Desktop\\college work\\Sem 5\\ai\\Project\\intents.json").read()
intents=json.loads(data_file)

for intent in intents['intents']:
    for pattern in intent['patterns']:
        w=nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w,intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
            
words=[lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore]
words=sorted(list(set(words)))
classes=sorted(list(set(classes)))

pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

#training data
training=[]
output_empty=[0]*len(classes)

for doc in documents:
    bag=[]
    pattern=doc[0]
    pattern=[ lemmatizer.lemmatize(word.lower()) for word in pattern ]
    for word in words:
        if word in pattern:
            bag.append(1)
        else:
            bag.append(0)
    output_row=list(output_empty)
    output_row[classes.index(doc[1])]=1
    training.append([bag,output_row])
    
random.shuffle(training)
training=np.array(training)  
X_train=list(training[:,0])
y_train=list(training[:,1])  

#Model
model=Sequential()
model.add(Dense(128,activation='relu',input_shape=(len(X_train[0]),)))
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(y_train[0]),activation='softmax'))

adam=Adam(0.001)
model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])

weights=model.fit(np.array(X_train),np.array(y_train),epochs=125,batch_size=15,verbose=1)    
model.save('leaf.h5',weights)

from keras.models import load_model
model = load_model('leaf.h5')
intents = json.loads(open("C:\\Users\\chinm\\OneDrive\\Desktop\\college work\\Sem 5\\ai\\Project\\intents.json").read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

#Predict
def clean_up(sentence):
    sentence_words=nltk.word_tokenize(sentence)
    sentence_words=[ lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def create_bow(sentence,words):
    sentence_words=clean_up(sentence)
    bag=list(np.zeros(len(words)))
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence,model):
    p=create_bow(sentence,words)
    res=model.predict(np.array([p]))[0]
    threshold=0.8
    results=[[i,r] for i,r in enumerate(res) if r>threshold]
    results.sort(key=lambda x: x[1],reverse=True)
    return_list=[]
    for result in results:
        return_list.append({'intent':classes[result[0]],'prob':str(result[1])})
    return return_list

def get_response(return_list,intents_json):
    print(return_list)
    if len(return_list)==0:
        tag='noanswer'
    else:    
        tag=return_list[0]['intent']

    list_of_intents= intents_json['intents']    
    for i in list_of_intents:
        if tag==i['tag'] :
            result= random.choice(i['responses'])
    return result

def response(text):
    return_list=predict_class(text,model)
    response=get_response(return_list,intents)
    return response

print(model.summary())

plt.plot(weights.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.savefig('accuracy.png')
plt.clf()

plt.plot(weights.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.savefig('loss.png')