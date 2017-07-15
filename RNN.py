import numpy as np
import random

# read file
file = open('input.txt', 'r')
data = file.read()
char = list(set(data))


VOCAB_SIZE = len(char)
DATA_LENGTH = len(data)

file.close()

char_to_index = {char:ix for ix,char in enumerate(char)}
index_to_char = {ix:char for ix,char in enumerate(char)}

#start creating neural network
class Network():

    def __init__(self, num_hidden_units, VOCAB_SIZE):
        self.num_hidden_units = num_hidden_units
        self.VOCAB_SIZE = VOCAB_SIZE
        self.parameters()


    def parameters(self):
        self.Wxh = np.random.randn(self.num_hidden_units, self.VOCAB_SIZE)*0.01 #input to hidden
        self.Whh = np.random.randn(self.num_hidden_units, self.num_hidden_units) * 0.01 #hidden to hidden
        self.Why = np.random.randn(self.VOCAB_SIZE, self.num_hidden_units) *0.01 #hidden to output
        self.bh = np.zeros((self.num_hidden_units, 1)) # bias hidden
        self.by = np.zeros((self.VOCAB_SIZE, 1)) # bias output


    #sequence_seed is the index of the character
    def sample(self, sequence_seed, length):

        newText = []
        h = np.zeros_like(self.bh)
        currentCharIndex = sequence_seed
        for t in range(0, length):
            x = np.zeros((self.VOCAB_SIZE, 1))
            x[currentCharIndex] = 1

            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)

            y = np.dot(self.Why, h) + self.by
            p = np.exp(y)/np.sum(np.exp(y))

            currentCharIndex = np.random.choice(range(VOCAB_SIZE), p=p.ravel())

            newChar = index_to_char[currentCharIndex]
            newText.append(newChar)

        return newText





    def lossFunc(self, inputs, targets, hprev):
        x_state, y_state, h_state, p_state = {}, {}, {}, {}
        loss = 0
        h_state[-1] = np.copy(hprev)

        #forward propagation
        for t in range(0, len(inputs)):
            # initialize inputs
            x_state[t] = np.zeros((self.VOCAB_SIZE, 1))
            x_state[t][inputs[t]] = 1

            # initialize hidden
            h_state[t] = np.tanh( np.dot(self.Wxh, x_state[t]) + np.dot(self.Whh, h_state[t-1])+ self.bh )

            #initialize output
            y_state[t] = np.dot(self.Why, h_state[t]) + self.by

            #set softmax
            p_state[t] = np.exp(y_state[t])/np.sum(np.exp(y_state[t]))
            loss += -np.log(p_state[t][targets[t],0])


        #initialize derivatives of weights
        dWhy, dWhh, dWxh, dbh, dby = np.zeros_like(self.Why), np.zeros_like(self.Whh), np.zeros_like(self.Wxh), np.zeros_like(self.bh), np.zeros_like(self.by)
        dhNext = np.zeros_like(h_state[0])
        #backward propagation
        for t in reversed(range(0, len(inputs))):
            dy = np.copy(p_state[t])
            #output backward propagation
            dy[targets[t]] -= 1 # find the error
            dWhy += np.dot(dy, h_state[t].T)
            dby += dy

            #hidden backward propagation
            dh = np.dot(self.Why.T, dy) + dhNext
            dhRaw = (1 - h_state[t]*h_state[t]) * dh
            dbh += dhRaw

            dWxh += np.dot(dhRaw, x_state[t].T)
            dWhh += np.dot(dhRaw, h_state[t-1].T)

            # set the next hidden delta
            dhNext = np.dot(self.Whh.T, dhRaw)


        # clip to prevent exploding gradient
        for dparam in [dWhy, dWhh, dWxh, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)

        return loss, dWxh, dWhh, dWhy, dbh, dby, h_state[len(inputs)-1] # return last time step of hidden step



    # put in the seq length not all the data
    def train(self):
        #memory
        mxh, mhh, mhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        mbh, mby = np.zeros_like(self.bh), np.zeros_like(self.by)
        #learning rate
        self.epsi = 1e-8
        learning_rate = 0.01

        hprev = np.zeros_like(self.bh)
        num_times = 0
        currentIndex =0
        batch_size = 32
        smooth_loss = -np.log(1.0 / self.VOCAB_SIZE) * batch_size

        hprev = np.zeros_like(self.bh)


        while(currentIndex+batch_size+1 < len(data)):
            currentIndex+= batch_size
            X = [char_to_index[char] for char in data[batch_size*num_times:currentIndex]]
            y = [char_to_index[char] for char in data[batch_size*num_times+1:currentIndex+1]]

            loss, dWxh, dWhh, dWhy, dbh, dby, hprev = self.lossFunc(X, y, hprev)

            smooth_loss = smooth_loss * 0.999 + loss * 0.001

            for param, dparam, mem in zip([self.Wxh, self.Whh, self.Why, self.bh, self.by],
                                          [dWxh, dWhh, dWhy, dbh, dby],
                                          [mxh, mhh, mhy, mbh, mby]):

                mem += dparam * dparam
                param += -learning_rate * dparam /np.sqrt(mem+self.epsi)

            num_times += 1

            if(num_times % 1000 == 0):
                seed = random.randint(0, self.VOCAB_SIZE-1)
                returnedValue = self.sample(seed, 100)
                print(''.join(returnedValue))
                print('---\nloss: %s\n---' % (loss,))




rnnModel = Network(100, VOCAB_SIZE)


print('start training...')

rnnModel.train()





