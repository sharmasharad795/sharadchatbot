import numpy as np
import tensorflow as tf
import re
import time
import string

#import the data

lines=open('movie_lines.txt',encoding='utf-8',errors='ignore').read().split('\n')
conversations=open('movie_conversations.txt',encoding='utf-8',errors='ignore').read().split('\n')

#mapping each line to its id using dictionary
id2line={}
#getting the id and its associated movie line
for line in lines:
    _line=line.split(' +++$+++ ')
    if len(_line) == 5:
        id2line[_line[0]]=_line[4]

#creating a list for list no for conversations
conversations_ids=[]

for conversation in conversations[:-1]:
    _conversation=conversation.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
    conversations_ids.append(_conversation.split(','))
        
#gettijng qs and ans
questions = []
answers = []    

#assigns questions and answers for each conversation.. there might be common for both qs and ans as ans to one might become qs to another while conversing
for conversation in conversations_ids:
    for x in range(len(conversation)-1):
        questions.append(id2line[conversation[x]])
        answers.append(id2line[conversation[x+1]])
        
#doing a cleanup of data for trainig chatbot
def clean_text(text):
        text=text.lower()
        text=re.sub(r"i'm","i am",text)
        text=re.sub(r"he's","he is",text)
        text=re.sub(r"she's","she is",text)
        text=re.sub(r"that's","that is",text)
        text=re.sub(r"what's","what is",text)
        text=re.sub(r"where's","where is",text)
        text=re.sub(r"\'ll"," will",text)
        text=re.sub(r"\'ve"," have",text)
        text=re.sub(r"\'re"," are",text)
        text=re.sub(r"\'d"," would",text)
        text=re.sub(r"won't","will not",text)
        text=re.sub(r"can't","cannot",text)
        text=re.sub(r"[-()\"#/@;:<>{}+=~|.?,]","",text)
        return text

clean_questions=[]
for question in questions:
    clean_questions.append(clean_text(question))
    
clean_answers=[]
for answer in answers:
    clean_answers.append(clean_text(answer))
    
#dictionary to map no of occurences of words
word2count={}
for question in clean_questions:
    for word in question.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1
            
for answer in clean_answers:
    for word in answer.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1            
     
#create dicts to map qs and ans words to a unique integer
threshold=20
#the higher the threshold,the lesser words we'll get and lesser the training time,and vice versa
questionswords2int={}
word_number=0
#checks if the word is above word threshold limit, if yes it is assigned a unique integer and added to the two lists 
for word,count in word2count.items():
    if count >= threshold:
        questionswords2int[word]=word_number
        word_number += 1
        
answerswords2int={}
word_number=0
for word,count in word2count.items():
    if count >= threshold:
        answerswords2int[word]=word_number
        word_number += 1

#adding the last tokens to the above two dictionaries
tokens=['<PAD>','<EOS>','<OUT>','<SOS>']
for token in tokens:
           questionswords2int[token]=len(questionswords2int)+1 
    
for token in tokens:
           answerswords2int[token]=len(answerswords2int)+1 
    
#creating inverse dict for answerswords2int
answersint2words={w_i:w for w,w_i in answerswords2int.items()}    

#adding EOS to clean_answers:
for i in range(len(clean_answers)):
    clean_answers[i] += ' <EOS>'
    
#translating all qs and ans into integers as in the dict
#and replacing all the words that were filtered out by <OUT>   
questions_into_int=[]
#checks if word is there in cleaN-qs and clean_ans list,if yes, they are included in respective qs and answer lists in the form of the unique integer they were assigned earlier
for question in clean_questions:
    ints=[]
    for word in question.split():
        if word not in questionswords2int:
            ints.append(questionswords2int['<OUT>'])
        else:
            ints.append(questionswords2int[word])
    questions_into_int.append(ints)

answers_into_int=[]
for answer in clean_answers:
    ints=[]
    for word in answer.split():
        if word not in answerswords2int:
            ints.append(answerswords2int['<OUT>'])
        else:
            ints.append(answerswords2int[word])
    answers_into_int.append(ints)

            
#sorting the above two list according to their length
sorted_clean_questions=[]
sorted_clean_answers=[]

for length in range(1, 25 + 1):
    for i in enumerate(questions_into_int):
        if len(i[1]) == length:
            sorted_clean_questions.append(questions_into_int[i[0]])
            sorted_clean_answers.append(answers_into_int[i[0]])
                        
#####PART2 - SEQ2SEQ arch - building it ###
            
##creating placeholders/varibales in tensorflow for the inputs (sorted cleam qs) and targets( actual ans- sorted cln ans)

def model_inputs():
    inputs = tf.placeholder(tf.int32, [None, None], name = 'input')
    targets = tf.placeholder(tf.int32, [None, None], name = 'target')
    lr = tf.placeholder(tf.float32, name = 'learning_rate')
    keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
    return inputs, targets, lr, keep_prob


#preprocessing the targets..since neural netwrok will accept answers in batches..ie lists of answers
def preprocess_targets(targets,word2int,batch_size):
#left_side adds a SOS token to the beginning of every qs and and list
#right_side gets all elements till the given size except the last element of each row of batch size to accomodate for the added SOS token, as we need to keep the same size 
#the two sides are then concatenated to get all rows till given batch size, and a new SOS col and all colms except last(or last element of each row)
    left_side=tf.fill([batch_size,1],word2int['<SOS>'])
    right_side=tf.strided_slice(targets,[0,0],[batch_size,-1],[1,1])
    preprocess_targets=tf.concat([left_side,right_side],1)
    return preprocess_targets
            
               
#creating the encoder RNN 
#lstm creates lstms ..rnn_size gives the no of lstms cells we want to create
#lstm_dropout controls the no of lstms we dont want to be active for trainig,ie their weights are excluded    
#rnn_inputs refers to model_input
#seq_length is listt of length of each qs in the batch
# decoder/enocer cells contain stacked lstm layers with dropout applied 
    
def encoder_rnn(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
    encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
    encoder_output, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = encoder_cell,
                                                                    cell_bw = encoder_cell,
                                                                    sequence_length = sequence_length,
                                                                    inputs = rnn_inputs,
                                                                    dtype = tf.float32)
    return encoder_state


#decoding the training set
#decoder_embedded_inputs are inputs on which embedding is done..
#embedding is a mapping of objects, such as words,  to vectors of real numbers..
#decoding scope is like an object of tf.variable_scope in tensorflow
#output_fn is used to return the decoder output at the end
# tf.zeroes in attn_states creates a 3d zero matrix   [no of lines,no of cols, no of elemenst]
#att_keys are keyss to be compared with target state
#att_values used to create the context vector..
#context vec is returned by encoder and should be used by encoder while decoding.
#att_score_fn used to compute similarity bw keys and target states
#att_const_fn to build attention state
#encoder_state[0] to get the value of encoder_state 
#only decoder_output is needed. dec_final_state and dec_final_context_state can also be done like _,_    
def decode_training_set(encoder_state, decoder_cell, decoder_embedded_input, sequence_length, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = "bahdanau", num_units = decoder_cell.output_size)
    training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              name = "attn_dec_train")
    decoder_output, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                              training_decoder_function,
                                                                                                              decoder_embedded_input,
                                                                                                              sequence_length,
                                                                                                              scope = decoding_scope)
    decoder_output_dropout = tf.nn.dropout(decoder_output, keep_prob)
    return output_function(decoder_output_dropout)
 



# Decoding the test/validation set
# we do something that we did above..just that now we docde the test set once training is done
#validation set is approx 10% of training set which is left for cross validation while working with test set
#in this function tf.contrib.seq2seq.attention_decoder_fn_inference is used instead of tf.contrib.seq2seq.attention_decoder_fn_train
#sos_id {sos token id]},eos_id{eos token id},max_length equals length of longest ans in batch
#num_words gives the count of all words of all ans..it actually refers to the count of answerwords2int dict
# so these are the 4 new arguments due to the new fn being used : tf.contrib.seq2seq.attention_decoder_fn_inference
#attention parameters are kept cuz they play an imp role inpredicting the final outcome
#decoder_embeddings_matrix is used instead of decoder_embeddings_inout    
def decode_test_set(encoder_state, decoder_cell, decoder_embeddings_matrix, sos_id, eos_id, maximum_length, num_words, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = "bahdanau", num_units = decoder_cell.output_size)
    test_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(output_function,
                                                                              encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              decoder_embeddings_matrix,
                                                                              sos_id,
                                                                              eos_id,
                                                                              maximum_length,
                                                                              num_words,
                                                                              name = "attn_dec_inf")
    test_predictions, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                                test_decoder_function,
                                                                                                                scope = decoding_scope)
    return test_predictions


# Creating the Decoder RNN
#encoder state is o/p of encoder which becomes i/p of decoder
#num_layers is the  no of layers we want inside decoder
#decoder/enocer cells contain stacked lstm layers with dropout applied    
#structure wise in RNN, we first have lstm,then stacked lstm layers,and lastly fully connected layers
#output_function creates the fully connected layer here
def decoder_rnn(decoder_embedded_input, decoder_embeddings_matrix, encoder_state, num_words, sequence_length, rnn_size, num_layers, word2int, keep_prob, batch_size):
    with tf.variable_scope("decoding") as decoding_scope:
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
        decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
        weights = tf.truncated_normal_initializer(stddev = 0.1)
        biases = tf.zeros_initializer()
        output_function = lambda x: tf.contrib.layers.fully_connected(x,
                                                                      num_words,
                                                                      None,
                                                                      scope = decoding_scope,
                                                                      weights_initializer = weights,
                                                                      biases_initializer = biases)
        training_predictions = decode_training_set(encoder_state,
                                                   decoder_cell,
                                                   decoder_embedded_input,
                                                   sequence_length,
                                                   decoding_scope,
                                                   output_function,
                                                   keep_prob,
                                                   batch_size)
        decoding_scope.reuse_variables()
        test_predictions = decode_test_set(encoder_state,
                                           decoder_cell,
                                           decoder_embeddings_matrix,
                                           word2int['<SOS>'],
                                           word2int['<EOS>'],
                                           sequence_length - 1,
                                           num_words,
                                           decoding_scope,
                                           output_function,
                                           keep_prob,
                                           batch_size)
    return training_predictions, test_predictions



#Building th3 seq2seq model
#inputs are the qs we already have,same for targets in terms of answers
#



def seq2seq_model(inputs, targets, keep_prob, batch_size, sequence_length, answers_num_words, questions_num_words, encoder_embedding_size, decoder_embedding_size, rnn_size, num_layers, questionswords2int):
    encoder_embedded_input = tf.contrib.layers.embed_sequence(inputs,
                                                              answers_num_words + 1,
                                                              encoder_embedding_size,
                                                              initializer = tf.random_uniform_initializer(0, 1))
    encoder_state = encoder_rnn(encoder_embedded_input, rnn_size, num_layers, keep_prob, sequence_length)
    preprocessed_targets = preprocess_targets(targets, questionswords2int, batch_size)
    decoder_embeddings_matrix = tf.Variable(tf.random_uniform([questions_num_words + 1, decoder_embedding_size], 0, 1))
    decoder_embedded_input = tf.nn.embedding_lookup(decoder_embeddings_matrix, preprocessed_targets)
    training_predictions, test_predictions = decoder_rnn(decoder_embedded_input,
                                                         decoder_embeddings_matrix,
                                                         encoder_state,
                                                         questions_num_words,
                                                         sequence_length,
                                                         rnn_size,
                                                         num_layers,
                                                         questionswords2int,
                                                         keep_prob,
                                                         batch_size)
    return training_predictions, test_predictions

##training  the seq2seq model#####


#epoch is one full iteration..one full forward and backward propagation
#num_layers is no of layers in encoder or decoder rnn
#embedding_size is no of cols in embedding matrix ie no of values we want...
#each line corresponds to each token in the qs corpus
#at test time neurons are always present, no dropout... dropout pplied only in training
#keep prob is 1- dropout rate..
#keep prob is done for hidden units not for input units

    
epochs = 100
batch_size = 64
rnn_size = 512
num_layers = 3
encoding_embedding_size = 512
decoding_embedding_size = 512
learning_rate = 0.01
learning_rate_decay = 0.9
min_learning_rate = 0.0001
keep_probability = 0.5
 
# Defining a session
#resetting the default tesnorflow graph
tf.reset_default_graph()
session = tf.InteractiveSession()
 
# Loading the model inputs
inputs, targets, lr, keep_prob = model_inputs()
 
# Setting the sequence length
#seq length gives the no of words in each qs or ans
#in sorted_clean_qs/ams we had taken max seq length to be 25

sequence_length = tf.placeholder_with_default(25, None, name = 'sequence_length')
 
# Getting the shape of the inputs tensor
#input_shape is used to give shape to a tensor 
input_shape = tf.shape(inputs)
 
# Getting the training and test predictions
#not the actaul test and training yet
#reverse is like reshape of numpy,,used to reshape the tensor
training_predictions, test_predictions = seq2seq_model(tf.reverse(inputs, [-1]),
                                                       targets,
                                                       keep_prob,
                                                       batch_size,
                                                       sequence_length,
                                                       len(answerswords2int),
                                                       len(questionswords2int),
                                                       encoding_embedding_size,
                                                       decoding_embedding_size,
                                                       rnn_size,
                                                       num_layers,
                                                       questionswords2int)

#Setting up the Loss Error, the Optimizer and Gradient Clipping
#grad clipping caps the gradients bw a min and a max value to prevent vanishing or exploding gradient issues by applying grad clipping to optimizer
#loss error based on wighted cross entropy loss error, most relevant while dealing with dnlp
#loss error measured bw training predictions and targets
#tf.ones takes a tensor of ones..first no rows and then no of cols..in args
with tf.name_scope("optimization"):
    loss_error = tf.contrib.seq2seq.sequence_loss(training_predictions,
                                                  targets,
                                                  tf.ones([input_shape[0], sequence_length]))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradients = optimizer.compute_gradients(loss_error)
    clipped_gradients = [(tf.clip_by_value(grad_tensor, -5., 5.), grad_variable) for grad_tensor, grad_variable in gradients if grad_tensor is not None]
    optimizer_gradient_clipping = optimizer.apply_gradients(clipped_gradients)
 
#Padding the sequences with the <PAD> token
#we are doing padding since we want alll sentences in batch irrespective of qs or ans must have same length
#padding helps to get length of qs seq = len of ans seq 
# logic is to take largest seq and fill others with pad tokens so that they are of same lenght
#max_sq_length is max length for all seq in bacth    
#word2int can be ether qs/answord2int dictionaries
    
def apply_padding(batch_of_sequences, word2int):
    max_sequence_length = max([len(sequence) for sequence in batch_of_sequences])
    return [sequence + [word2int['<PAD>']] * (max_sequence_length - len(sequence)) for sequence in batch_of_sequences]
 
# Splitting the data into batches of questions and answers
#qs is inputs, ans are targets
#start index is index of first qs/ans adding to the batch..
# no of qs=no of ans..thus qs/batch_size will give no of batches
#batch_index gives a index no to each batch..(0,1,2..etc..)
#start_index starts from 0,then 64 etc..if batch size is 64...
#yield is same as return but its bttr for sequences
def split_into_batches(questions, answers, batch_size):
    for batch_index in range(0, len(questions) // batch_size):
        start_index = batch_index * batch_size
        questions_in_batch = questions[start_index : start_index + batch_size]
        answers_in_batch = answers[start_index : start_index + batch_size]
        padded_questions_in_batch = np.array(apply_padding(questions_in_batch, questionswords2int))
        padded_answers_in_batch = np.array(apply_padding(answers_in_batch, answerswords2int))
        yield padded_questions_in_batch, padded_answers_in_batch
 

# Splitting the questions and answers into training and validation sets
#len(sorted_clean_qs)=len(sorted_clean_ans)..thus we're using just one tr_valid_split
training_validation_split = int(len(sorted_clean_questions) * 0.15)
training_questions = sorted_clean_questions[training_validation_split:]
training_answers = sorted_clean_answers[training_validation_split:]
validation_questions = sorted_clean_questions[:training_validation_split]
validation_answers = sorted_clean_answers[:training_validation_split]
 
# Training
#batch_index_check_training_loss will check training loss after every 100 batches
batch_index_check_training_loss = 100
batch_index_check_validation_loss = ((len(training_questions)) // batch_size // 2) - 1
total_training_loss_error = 0
list_validation_loss_error = []
early_stopping_check = 0
early_stopping_stop = 1000
checkpoint = "chatbot_weights.ckpt" 
session.run(tf.global_variables_initializer())
for epoch in range(1, epochs + 1):
    for batch_index, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(training_questions, training_answers, batch_size)):
        starting_time = time.time()
        _, batch_training_loss_error = session.run([optimizer_gradient_clipping, loss_error], {inputs: padded_questions_in_batch,
                                                                                               targets: padded_answers_in_batch,
                                                                                               lr: learning_rate,
                                                                                               sequence_length: padded_answers_in_batch.shape[1],
                                                                                               keep_prob: keep_probability})
        total_training_loss_error += batch_training_loss_error
        ending_time = time.time()
        batch_time = ending_time - starting_time
        if batch_index % batch_index_check_training_loss == 0:
            print('Epoch: {:>3}/{}, Batch: {:>4}/{}, Training Loss Error: {:>6.3f}, Training Time on 100 Batches: {:d} seconds'.format(epoch,
                                                                                                                                       epochs,
                                                                                                                                       batch_index,
                                                                                                                                       len(training_questions) // batch_size,
                                                                                                                                       total_training_loss_error / batch_index_check_training_loss,
                                                                                                                                       int(batch_time * batch_index_check_training_loss)))
            total_training_loss_error = 0
        if batch_index % batch_index_check_validation_loss == 0 and batch_index > 0:
            total_validation_loss_error = 0
            starting_time = time.time()
            for batch_index_validation, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(validation_questions, validation_answers, batch_size)):
                batch_validation_loss_error = session.run(loss_error, {inputs: padded_questions_in_batch,
                                                                       targets: padded_answers_in_batch,
                                                                       lr: learning_rate,
                                                                       sequence_length: padded_answers_in_batch.shape[1],
                                                                       keep_prob: 1})
                total_validation_loss_error += batch_validation_loss_error
            ending_time = time.time()
            batch_time = ending_time - starting_time
            average_validation_loss_error = total_validation_loss_error / (len(validation_questions) / batch_size)
            print('Validation Loss Error: {:>6.3f}, Batch Validation Time: {:d} seconds'.format(average_validation_loss_error, int(batch_time)))
            learning_rate *= learning_rate_decay
            if learning_rate < min_learning_rate:
                learning_rate = min_learning_rate
            list_validation_loss_error.append(average_validation_loss_error)
            if average_validation_loss_error <= min(list_validation_loss_error):
                print('I speak better now!!')
                early_stopping_check = 0
                saver = tf.train.Saver()
                saver.save(session, checkpoint)
            else:
                print("Sorry I do not speak better, I need to practice more.")
                early_stopping_check += 1
                if early_stopping_check == early_stopping_stop:
                    break
    if early_stopping_check == early_stopping_stop:
        print("My apologies, I cannot speak better anymore. This is the best I can do.")
        break
print("Game Over")






























