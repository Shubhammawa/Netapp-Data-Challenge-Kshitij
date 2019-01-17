
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from nltk.tokenize import RegexpTokenizer
import math
import gensim
from gensim.models import Word2Vec
from gensim.models import FastText


# In[2]:


from tensorflow.python.framework import ops


# In[3]:


data = pd.read_csv("train.csv",dtype=object,na_values=str).values


# In[4]:


x = np.array(data[:,2:4])
y = np.array(data[:,1])
print(x[0:4])


# In[5]:


#X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33)


# In[6]:


corpus_raw = []
for i in range(0,x.shape[0]):
    for sent in x[i]:
        if np.nan_to_num(sent) == 0:     # If short description is missing
            corpus_raw.append("NA")      # Necessary, otherwise while feeding training data into NN can cause mismatch
        elif np.nan_to_num(sent) != 0:
            corpus_raw.append(sent)


# In[7]:


print(corpus_raw[0:5])
print(np.shape(corpus_raw))
print(type(corpus_raw[24199]))
#words = corpus_raw[0].split()
#print(words)
print(np.nan_to_num(corpus_raw[24199]))


# In[8]:


# List of words
words2 = []
tokenizer = RegexpTokenizer(r'\w+')
i = 0
for sent in corpus_raw:
    #words.append(sent.split())
    #i = i + 1
    #print(i)
    for word in tokenizer.tokenize(sent):
        words2.append(word.lower())
    
# for i in range(0,len(corpus_raw)):
#     for word in corpus_raw[i].split():
#         if word != '.':       # because we don't want to treat . as a word
#             words.append(word)
#print(type(words))
#words = np.array(words)
#print(type(words))
#print(len(words))
#print(words[0])

Vocab = set(words2)    # so that all duplicate words are removed
vocab = list(Vocab)
# vocab = []
# for j in words:
#     if j not in vocab:
#         vocab.append(j)


word2int = {}
int2word = {}
vocab_size = len(vocab)  # gives the total number of unique words

for i,word in enumerate(vocab):
    word2int[word] = i
    int2word[i] = word


# In[9]:


# List of lists of words
words = []
sentences = []
tokenizer = RegexpTokenizer(r'\w+')
i = 0
for sent in corpus_raw:
    #words.append(sent.split())
    #i = i + 1
    #print(i)
    for word in tokenizer.tokenize(sent):
        words.append(word.lower())
    sentences.append(words)
    words = []
# for i in range(0,len(corpus_raw)):
#     for word in corpus_raw[i].split():
#         if word != '.':       # because we don't want to treat . as a word
#             words.append(word)
#print(type(words))
#words = np.array(words)
#print(type(words))
#print(len(words))
#print(words[0])

#Vocab = set(words)    # so that all duplicate words are removed
#vocab = list(Vocab)
# vocab = []
# for j in words:
#     if j not in vocab:
#         vocab.append(j)


# word2int = {}
# int2word = {}
# vocab_size = len(vocab)  # gives the total number of unique words

# for i,word in enumerate(vocab):
#     word2int[word] = i
#     int2word[i] = word


# In[10]:


print(len(words))
print(vocab_size)
print(words[0:100])
print(type(vocab))
#Vocab = list(vocab)
#print(vocab[0:100])
# for i in range(0,5):
#     print(i)
# for sent in corpus_raw:
#     print(type(sent))
print(len(word2int))
print(len(int2word))
#print(int2word)
print(len(sentences))
print(sentences[0:5])


# In[11]:


sentences = np.array(sentences)


# In[12]:


#model_word2vec = Word2Vec(sentences, size=100, window=5, min_count=0,workers=10,sg=0)
#model_word2vec.train(sentences,total_examples=len(sentences),epochs=10)


model = Word2Vec(sentences, size=100, window=5, min_count=0,workers=10,sg=0)
model.train(sentences,total_examples=len(sentences),epochs=10)

# In[13]:


#print(model_word2vec)
#print(model_word2vec['killing'])
#print(model_word2vec.wv.most_similar("shootings"))


# In[14]:


#model_word2vec.save("Saved_model_word2vec")


# In[12]:


#model = Word2Vec.load("Saved_model_word2vec")
#print(model)


# In[13]:


embeddings = model[model.wv.vocab]


# In[14]:


print(type(embeddings))
print(embeddings.shape)


# In[15]:


print(embeddings[2])
print(model['there'])
#print(model['There'])


# In[16]:


# #print(word2int[0])
# for i in range(0,10):
#     print(int2word[i])


# In[17]:


#word_vectors = np.zeros([],dtype=float)
#print(words2[0:100])
print(sentences[1])
print(len(sentences[1]))
#print(type(sentences))


# In[18]:


print(model[sentences[1]])
print(0%2)
print(1%2)
print(model['she'])


# In[22]:


X = []
Y = []
temp = []
for i in range(0,len(sentences)):
    for j in range(0,len(sentences[i])):
        temp.append(model[sentences[i][j]])
    if(i%2!=0):
        X.append(temp)
        temp = []
        


# In[23]:


print(type(X))
print(np.shape(X))


# In[24]:


print(len(X[0]))    # Should be sum of no of words in sentences 1 and 2 (Headline and description for first training example)
# Output = 27 (14+13)  Training data ready.


# In[25]:


# test = []
# test2 = []
# test.append(model[sentences[0][0]])
# print(test)
# test.append(model[sentences[0][1]])
# print(test)
# test2.append(test)
# print(test2)


# In[26]:


print(len(X[3443]))
max1 = 0
for i in range(0,len(X)):
    if(len(X[i])>max1):
        max1 = len(X[i])
        pos = i
print(max1)
print(pos)


# In[27]:


print(58142*2)


# In[28]:


print(len(sentences[116285]))


# In[29]:


count = 0
for i in range(0,len(X)):
    if(len(X[i])>100):
        count = count + 1
print(count)

#  Sequence length : Length of each training example
#  Sequence length is varying from 1 to 250, we have to choose a dimension and accordingly all training exapmles would be
#  padded or truncated


# In[30]:


import keras
from keras.preprocessing.sequence import pad_sequences
X_new = keras.preprocessing.sequence.pad_sequences(sequences=X, maxlen=100, dtype='float32', padding='post', truncating='post', value=0.0)


# In[37]:


#print(X[0])
#print(X_new[0])
#print(len(X_new[0]))


# In[60]:


print(type(X_new))
print(np.shape(X_new))


# In[39]:


#np.save("X_new",X_new)


# In[19]:


#X = np.load("/media/shubham/1A2A3CBF2A3C99A9/Academics/Self/Netapp-Data-Challenge-Kshitij-storage/X_new.npy")


# In[21]:


#np.savetxt("X_new_text",X_new)


# In[22]:


###-----------------One-hot labels generation------------------###


# In[20]:


print(len(y))
print(y[0:4])
print(list(set(y)))
print(len(list(set(y))))


# In[21]:


labels = list(set(y))


# In[22]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(labels)
y_labels = le.transform(y)


# In[23]:


#print(y_labels[0:20])
one_hot_labels = tf.keras.utils.to_categorical(y_labels)


# In[24]:


print(np.shape(one_hot_labels))
print(one_hot_labels[0])


# In[53]:


#import csv
#with open('labels.csv', 'w') as csvfile:
#    wr = csv.writer(csvfile)
#    wr.writerow(y_labels)


# In[54]:


#np.savetxt("one_hot_labels.csv", one_hot_labels, delimiter=",")


# In[ ]:


###--------------------------CNN model-------------------------###


# In[25]:


def create_placeholders(seq_length, embedding_size, n_y):
    
#     Creates the placeholders for the tensorflow session.
    
#     Arguments:
#     n_H0 -- scalar, height of an input image
#     n_W0 -- scalar, width of an input image
#     n_C0 -- scalar, number of channels of the input
#     n_y -- scalar, number of classes
        
#     Returns:
#     X -- placeholder for the data input, of shape [None, n_H0, n_W0, n_C0] and dtype "float"
#     Y -- placeholder for the input labels, of shape [None, n_y] and dtype "float"
    

    ### START CODE HERE ### (≈2 lines)
    X = tf.placeholder(dtype = tf.float32, shape=(None,seq_length,embedding_size,1))
    Y = tf.placeholder(dtype = tf.float32, shape=(None,n_y))
    ### END CODE HERE ###
    
    return X, Y


# In[26]:


# def initialize_parameters(filter_size,embedding_size,num_filters):
#     # Initializes weight parameters
#     W = tf.get_variable("W",[filter_size,embedding_size,1,num_filters],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer(seed=0),regularizer = tf.contrib.layers.l2_regularizer(scale=0.1))
#     return W

def initialize_parameters(filter_sizes,embedding_size,num_filters):
    # Initializes weight parameters
    W1 = tf.get_variable("W1",[filter_sizes[0],embedding_size,1,num_filters],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer(seed=0),regularizer = tf.contrib.layers.l2_regularizer(scale=0.1))
    W2 = tf.get_variable("W2",[filter_sizes[1],embedding_size,1,num_filters],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer(seed=0),regularizer = tf.contrib.layers.l2_regularizer(scale=0.1))
    W3 = tf.get_variable("W3",[filter_sizes[2],embedding_size,1,num_filters],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer(seed=0),regularizer = tf.contrib.layers.l2_regularizer(scale=0.1))
    W4 = tf.get_variable("W4",[filter_sizes[3],embedding_size,1,num_filters],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer(seed=0),regularizer = tf.contrib.layers.l2_regularizer(scale=0.1))
    
    parameters = {"W1": W1,
                  "W2": W2,
                  "W3": W3,
                  "W4": W4}
    
    return parameters


# In[27]:


# def forward_propagation(X,filter_sizes,embedding_size,num_filters,seq_length):
#     P2 = []
#     for filter_size in filter_sizes:
#         W = initialize_parameters(filter_size,embedding_size,num_filters)
#         Z = tf.nn.conv2d(X,W,strides=[1,1,1,1],padding="SAME")
#         A = tf.nn.relu(Z)
#         P = tf.nn.max_pool(A,ksize=[1,seq_length-filter_size+1,1,1],strides=[1,1,1,1],padding="SAME")
#         P2.append(P)
#     Z2 = tf.contrib.layers.fully_connected(P2,41,activation_fn = None)
#     return Z2

def forward_propagation(X,filter_sizes,embedding_size,num_filters,seq_length,parameters):
    P = []
    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']
    W4 = parameters['W4']
    #W1 = initialize_parameters(filter_sizes[0],embedding_size,num_filters)
    Z1 = tf.nn.conv2d(X,W1,strides=[1,1,1,1],padding="SAME")
    A1 = tf.nn.relu(Z1)
    P1 = tf.nn.max_pool(A1,ksize=[1,seq_length-filter_sizes[0]+1,1,1],strides=[1,1,1,1],padding="SAME")
    P.append(P1)
    
    #W2 = initialize_parameters(filter_sizes[1],embedding_size,num_filters)
    Z2 = tf.nn.conv2d(X,W2,strides=[1,1,1,1],padding="SAME")
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool(A2,ksize=[1,seq_length-filter_sizes[1]+1,1,1],strides=[1,1,1,1],padding="SAME")
    P.append(P2)
    
    #W3 = initialize_parameters(filter_sizes[2],embedding_size,num_filters)
    Z3 = tf.nn.conv2d(X,W3,strides=[1,1,1,1],padding="SAME")
    A3 = tf.nn.relu(Z3)
    P3 = tf.nn.max_pool(A3,ksize=[1,seq_length-filter_sizes[2]+1,1,1],strides=[1,1,1,1],padding="SAME")
    P.append(P3)
    
    #W4 = initialize_parameters(filter_sizes[3],embedding_size,num_filters)
    Z4 = tf.nn.conv2d(X,W4,strides=[1,1,1,1],padding="SAME")
    A4 = tf.nn.relu(Z4)
    P4 = tf.nn.max_pool(A4,ksize=[1,seq_length-filter_sizes[3]+1,1,1],strides=[1,1,1,1],padding="SAME")
    P.append(P4)
    
    Z5 = tf.contrib.layers.fully_connected(P4,41,activation_fn = None)
    return Z5


# In[28]:


def compute_cost(Z5, Y):
    """
    Computes the cost
    
    Arguments:
    Z5 -- output of forward propagation (output of the last LINEAR unit), of shape (number of examples,41)
    Y -- "true" labels vector placeholder, same shape as Z5
    
    Returns:
    cost - Tensor of the cost function
    """
    
    ### START CODE HERE ### (1 line of code)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z5, labels = Y))
    ### END CODE HERE ###
    
    return cost


# In[29]:


def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    np.random.seed(seed)            
    m = X.shape[1]                  # number of training examples
    mini_batches = []
        
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((41,m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k*mini_batch_size:(k+1)*mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k*mini_batch_size:(k+1)*mini_batch_size]
        
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:,num_complete_minibatches*mini_batch_size:m]
        mini_batch_Y = shuffled_Y[:,num_complete_minibatches*mini_batch_size:m]
        
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches


# In[30]:


def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.007,
          num_epochs = 120, minibatch_size = 64, print_cost = True):
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep results consistent (tensorflow seed)
    seed = 3                                          # to keep results consistent (numpy seed)
    
    ## To be used if not using stochastic
    #(m, seq_length, embedding_size,nc) = X_train.shape             
    ##-----------------------------------------###
    
    
    ## To be used if using Stochastic ##
    m = X_train.shape[0]
    seq_length = X_train.shape[2]
    embedding_size = X_train.shape[3]
    nc = X_train.shape[4]
    ##------------------------------------####
    
    
    
    n_y = Y_train.shape[2]            # 2 - stochastic;  1 - otherwise                            
    costs = []                                        # To keep track of the cost
    filter_sizes = [2,3,5,7]
    num_filters = 5
    # Create Placeholders of the correct shape
    X, Y = create_placeholders(seq_length, embedding_size, n_y)

    # Initialize parameters
    parameters = initialize_parameters(filter_sizes,embedding_size,num_filters)
    
    # Forward propagation: Build the forward propagation in the tensorflow graph

    Z5 = forward_propagation(X,filter_sizes,embedding_size,num_filters,seq_length,parameters)
    
    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(Z5, Y)
    
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer that minimizes the cost.
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    # Initialize all the variables globally
    init = tf.global_variables_initializer()
     
    # Start the session to compute the tensorflow graph
    with tf.device("/gpu:0"):
        with tf.Session() as sess:

            # Run the initialization
            sess.run(init)

            # Do the training loop
            for epoch in range(num_epochs):
                #_, temp_cost = sess.run([optimizer, cost], feed_dict = {X:X_train, Y:Y_train})  Batch Gradient Descent

    #             minibatch_cost = 0.
    #             num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
    #             seed = seed + 1
    #             minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

    #             for minibatch in minibatches:

    #                 # Select a minibatch
    #                 (minibatch_X, minibatch_Y) = minibatch
    #                 # IMPORTANT: The line that runs the graph on a minibatch.
    #                 # Run the session to execute the optimizer and the cost, the feedict should contain a minibatch for (X,Y).
    #                 ### START CODE HERE ### (1 line)
    #                 _ , temp_cost = sess.run([optimizer, cost], feed_dict = {X: minibatch_X, Y: minibatch_Y})     # mini_batch gradieent descent
    #                 ### END CODE HERE ###

    #                 minibatch_cost += temp_cost / num_minibatches
                stochastic_cost=0    
                for i in range(0,m):
                    _, temp_cost = sess.run([optimizer, cost], feed_dict = {X:X_train[i], Y:Y_train[i]}) 
                    stochastic_cost += temp_cost/m

                # Print the cost every epoch
    #             if print_cost == True and epoch % 5 == 0:
    #                 print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
    #             if print_cost == True and epoch % 1 == 0:
    #                 costs.append(minibatch_cost)
                if print_cost == True and epoch % 5 == 0:
                    print ("Cost after epoch %i: %f" % (epoch, stochastic_cost))
                if print_cost == True and epoch % 1 == 0:
                    costs.append(stochastic_cost)


            # plot the cost
            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iterations (per tens)')
            plt.title("Learning rate =" + str(learning_rate))
            plt.show()
            # Calculate the correct predictions
            predict_op = tf.argmax(Z5, 1)
            correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))

            # Calculate accuracy on the test set
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            print(accuracy)
            train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
            test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
            print("Train Accuracy:", train_accuracy)
            print("Test Accuracy:", test_accuracy)

            return train_accuracy, test_accuracy, predict_op,parameters


# In[31]:


X = X_new    # Comment out if loaded from X_new.npy, otherwise run
#Y = one_hot_labels
#print(np.shape(X))


# In[32]:


#print(np.shape(Y))
#print(X[0][0])
#Y = Y.T
#print(np.shape(Y))


# In[33]:


#print(type(X))
#print(type(Y))
Y = one_hot_labels


# In[34]:


#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)
# Giving memory error
X_train = X[0:150000]
X_test = X[150000:]
Y_train = Y[0:150000]
Y_test = Y[150000:]


# In[35]:


# Y_train = Y_train.T
# Y_test = Y_test.T


# In[36]:


X_train = np.expand_dims(X_train,axis=3)
X_test = np.expand_dims(X_test,axis=3)

X_train = np.expand_dims(X_train,axis=1)      #Only to be used, if using stochastic gradient descent
X_test = np.expand_dims(X_test,axis=1)        #Only to be used, if using stochastic gradient descent
Y_train = np.expand_dims(Y_train,axis=1)
Y_test = np.expand_dims(Y_test,axis=1)


# In[37]:


print(np.shape(X_train))
print(np.shape(Y_train))


# In[ ]:

with tf.device("/gpu:0"):
    _, _, predictions,parameters = model(X_train, Y_train, X_test, Y_test)

