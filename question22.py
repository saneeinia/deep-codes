# region library
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn

#region functions

def myDiag(x):
    s = x.shape
    out = np.zeros((s[1], s[1]))
    for i in range(s[1]):
        out[i, i] = x[0, i]
    return out

def myMSE(x):
    s = x.shape
    x = x*x
    summ = 0

    for i in range(s[0]):
        summ = summ + x[i, 0]
    return summ/s[0]

def normaliazation():
    global data
    mini = 0
    maxi = 0
    s = data.shape
    for i in range(s[0]):
        if maxi < data[i, 0]:
            maxi = data[i, 0]

        elif mini > data[i, 0]:
            mini = data[i, 0]
    for i in range(s[0]):
        data[i, 0] = ((data[i, 0] - mini) / (maxi - mini))

    return 0


def initializeData():
    global m
    global data
    global input_data

    normaliazation()

    for j in range(m-1):
        for i in range(data.size - mm):
            input_data[i, j] = data[i + j, 0]
    for i in range(data.size - mm):
        input_data[i, m-1] = data[i+mm-1, 0]
    # print(input_data)
    # print(input_data.shape)
    return 0


def myF(x):
    return sig(x)


def myFPrime(x):
    return sigPrime(x)


#endregion

# region activation functions
def tanh(x):
    return ((math.e**(x)) - (math.e**(-x)))/((math.e**(x)) + (math.e**(-x)))


def tanhPrime(x):
    return 1-(tanh(x)**2)


def sig(x):
    return 1 / (1 + (math.e**(-x)))



def sigPrime(x):
    return sig(x)*(1-sig(x))


def actFun(x):
   return sig(x)


def actFunPrime(x):
    return sigPrime(x)

# endregion

# region class AutoEncoder

class autoEncoder:
    loww = 1
    highh = -1
    eta = 0.1
    trainNum = 200

    def __init__(self,inp,hiddenNum):


        self.inp =inp
        self.hiddenNum =hiddenNum
        self.w1 = np.random.uniform(self.loww, self.highh, (self.hiddenNum, self.inp))
        self.w2 = np.random.uniform(self.loww, self.highh, (self.inp,self.hiddenNum))
        self.outp = np.zeros((inp,1))
        self.hiddenLay = np.zeros((self.hiddenNum, 1))
        self.error = np.zeros((inp,1))
        self.prime = np.zeros(self.outp.shape)
    def AF1(self, x):
        return sig(x)


    def AFP1(self, x):
        return sigPrime(x)

    def AF2(self, x):
        return x
        # return sig(x)

    def AFP2(self, x):
        return np.ones(x.shape)
        #  return sigPrime(x)

    def train(self, x):

        net1 = self.w1 @ x
        self.hiddenLay = self.AF1(net1)

        net2 = self.w2 @ self.hiddenLay
        self.    outp = self.AF2(net2)

        self.error = x-self.outp
        self.prime = self.AFP2(net2)
        self.w2 = self.w2 - self.eta * -1 * (self.error * self.prime) @ np.transpose(self.hiddenLay)
        self.w1 = np.transpose(self.w2)


    def trainLoop(self,x):
        for i in range(self.trainNum):
            self.train(x)
    def cal(self,x):
        net1 = self.w1 @ x
        return (self.AF1(net1))




# endregion


# region main code

# region prepare data
data = pd.read_excel('DLdata1.xlsx', header=None)
data = data.to_numpy()
m = 11
mm =15
train_rate = 0.75
input_data = np.zeros((data.size - mm, m))
initializeData()
# endregion



# region initialize  number of neurons in each layer
numberOfNeuronsInAE1 = 20
numberOfNeuronsInAE2 = 40
numberOfNeuronsInAE3 = 20
numberOfNeuronsInAE4 = 30
numberOfNeuronsInAE5 = 10
numberOfNeuronsInAE6 = 4


numberOfData = input_data.shape[0]
numberOfInput = numberOfNeuronsInAE6
numberOfNeuronsInSecondLayer = 6
numberOfNeuronsInThirdLayer = 8
numberOfNeuronsInForthLayer = 10
numberOfNeuronsInFifthLayer = 8
numberOfNeuronsInSixthLayer = 6
numberOfOutput = 1
numOfTrain = round(numberOfData*train_rate)
numOfTest = numberOfData-numOfTrain
# endregion

# region initialize weights

loww = -1
highh = 1
ae1 = autoEncoder(10,numberOfNeuronsInAE1)
ae2 = autoEncoder(numberOfNeuronsInAE1, numberOfNeuronsInAE2)
ae3 = autoEncoder(numberOfNeuronsInAE2, numberOfNeuronsInAE3)
ae4 = autoEncoder(numberOfNeuronsInAE3, numberOfNeuronsInAE4)
ae5 = autoEncoder(numberOfNeuronsInAE4, numberOfNeuronsInAE5)
ae6 = autoEncoder(numberOfNeuronsInAE5, numberOfNeuronsInAE6)

w1 = np.random.uniform(loww, highh, (numberOfNeuronsInSecondLayer, numberOfInput))
net1 = np.zeros((1, numberOfNeuronsInSecondLayer))
o1 = np.zeros((1, numberOfNeuronsInSecondLayer))

w2 = np.random.uniform(loww, highh, (numberOfNeuronsInThirdLayer, numberOfNeuronsInSecondLayer))
net2 = np.zeros((1, numberOfNeuronsInThirdLayer))
o2 = np.zeros((1, numberOfNeuronsInThirdLayer))

w3 = np.random.uniform(loww, highh, (numberOfNeuronsInForthLayer, numberOfNeuronsInThirdLayer))
net3 = np.zeros((1, numberOfNeuronsInForthLayer))
o3 = np.zeros((1, numberOfNeuronsInForthLayer))

w4 = np.random.uniform(loww, highh, (numberOfNeuronsInFifthLayer, numberOfNeuronsInForthLayer))
net4 = np.zeros((1, numberOfNeuronsInFifthLayer))
o4 = np.zeros((1, numberOfNeuronsInFifthLayer))

w5 = np.random.uniform(loww, highh, (numberOfNeuronsInSixthLayer, numberOfNeuronsInFifthLayer))
net5 = np.zeros((1, numberOfNeuronsInSixthLayer))
o5 = np.zeros((1, numberOfNeuronsInSixthLayer))

w6 = np.random.uniform(loww, highh, (numberOfOutput, numberOfNeuronsInSixthLayer))
net6 = np.zeros((1, numberOfOutput))
o6 = np.zeros((1, numberOfOutput))
# endregion

# region variables

eta = 0.1
maxOfEpoch = 500
train_error = np.zeros((numOfTrain, 1))
test_error = np.zeros((numOfTest, 1))
output_train = np.zeros((numOfTrain, 1))
output_test = np.zeros((numOfTest, 1))

mse_train = np.zeros((maxOfEpoch, 1))
mse_test = np.zeros((maxOfEpoch, 1))

myInput = np.zeros((1, m-1))
myInput2 = np.zeros((1,numberOfNeuronsInAE6))
myTarget = 0
error = 0
c = 0
# endregion

# region loop for epochs


# region train stackAutoEncoder
for i in range(maxOfEpoch):


    for j in range(numOfTrain):
        for k in range(m - 1):
            myInput[0, k] = input_data[j, k]
        ae1.train(np.transpose(myInput))
    for j in range(numOfTrain):
        for k in range(m - 1):
            myInput[0, k] = input_data[j, k]
        ae2.train(ae1.cal(np.transpose(myInput)))
    for j in range(numOfTrain):
        for k in range(m - 1):
            myInput[0, k] = input_data[j, k]
        ae3.train(ae2.cal(ae1.cal(np.transpose(myInput))))
    for j in range(numOfTrain):
        for k in range(m - 1):
            myInput[0, k] = input_data[j, k]
        ae4.train(ae3.cal(ae2.cal(ae1.cal(np.transpose(myInput)))))
    for j in range(numOfTrain):
        for k in range(m - 1):
            myInput[0, k] = input_data[j, k]
        ae5.train(ae4.cal(ae3.cal(ae2.cal(ae1.cal(np.transpose(myInput))))))
    for j in range(numOfTrain):
        for k in range(m - 1):
            myInput[0, k] = input_data[j, k]
        ae6.train(ae5.cal(ae4.cal(ae3.cal(ae2.cal(ae1.cal(np.transpose(myInput)))))))


# endregion

for i in range(maxOfEpoch):

    if i == 150:
        eta = 0.01
    if i ==200:
        eta = 0.005
    if i==250:
        eta == 0.001




    # region train network
    for j in range(numOfTrain):
        for k in range(m-1):
            myInput[0, k] = input_data[j, k]

        myTarget = input_data[j, m-1]
        myInput2 = np.transpose(ae6.cal(ae5.cal(ae4.cal(ae3.cal(ae2.cal(ae1.cal(np.transpose(myInput))))))))

        net1 = myInput2 @ np.transpose(w1)
        o1 = myF(net1)

        net2 = o1 @ np.transpose(w2)
        o2 = myF(net2)

        net3 = o2 @ np.transpose(w3)
        o3 = myF(net3)

        net4 = o3 @ np.transpose(w4)
        o4 = myF(net4)

        net5 = o4 @ np.transpose(w5)
        o5 = myF(net5)

        net6 = o5 @ np.transpose(w6)
        o6 = net6

        error = myTarget-o6

        c = myFPrime(o5)
        A = myDiag(c)

        c = myFPrime(o4)
        B = myDiag(c)

        c = myFPrime(o3)
        C = myDiag(c)

        c = myFPrime(o2)
        D = myDiag(c)

        c = myFPrime(o1)
        E = myDiag(c)

        w1 = w1 - eta * error * -1 * 1 * np.transpose(w6 @ A @ w5 @ B @ w4 @ C @ w3 @ D @ w2 @ E) @ myInput2
        w2 = w2 - eta * error * -1 * 1 * np.transpose(w6 @ A @ w5 @ B @ w4 @ C @ w3 @ D)@o1
        w3 = w3 - eta * error * -1 * 1 * np.transpose(w6 @ A @ w5 @ B @ w4 @ C)@o2
        w4 = w4 - eta * error * -1 * 1 * np.transpose(w6 @ A @ w5 @ B)@o3
        w5 = w5 - eta * error * -1 * 1 * np.transpose(w6 @ A)@o4
        w6 = w6 - eta * error * -1 * 1 * o5



    # endregion

    # region test train_data
    for j in range(numOfTrain):
        for k in range(m - 1):
            myInput[0, k] = input_data[j, k]

        myTarget = input_data[j, m - 1]
        myInput2 = np.transpose(ae6.cal(ae5.cal(ae4.cal(ae3.cal(ae2.cal(ae1.cal(np.transpose(myInput))))))))

        net1 = myInput2 @ np.transpose(w1)
        o1 = myF(net1)

        net2 = o1 @ np.transpose(w2)
        o2 = myF(net2)

        net3 = o2 @ np.transpose(w3)
        o3 = myF(net3)

        net4 = o3 @ np.transpose(w4)
        o4 = myF(net4)

        net5 = o4 @ np.transpose(w5)
        o5 = myF(net5)

        net6 = o5 @ np.transpose(w6)
        o6 = net6

        error = myTarget-o6

        train_error[j] = error

        output_train[j] = o6
    # endregion

    # region test test_data
    for j in range(numOfTest):

        for k in range(m - 1):
            myInput[0, k] = input_data[j+numOfTrain, k]

        myTarget = input_data[j+numOfTrain, m - 1]
        myInput2 = np.transpose(ae6.cal(ae5.cal(ae4.cal(ae3.cal(ae2.cal(ae1.cal(np.transpose(myInput))))))))

        net1 = myInput2 @ np.transpose(w1)
        o1 = myF(net1)

        net2 = o1 @ np.transpose(w2)
        o2 = myF(net2)

        net3 = o2 @ np.transpose(w3)
        o3 = myF(net3)

        net4 = o3 @ np.transpose(w4)
        o4 = myF(net4)

        net5 = o4 @ np.transpose(w5)
        o5 = myF(net5)

        net6 = o5 @ np.transpose(w6)
        o6 = net6

        error = myTarget-o6

        test_error[j] = error

        output_test[j] = o6
    # endregion

    mse_train[i] = myMSE(train_error)
    mse_test[i] = myMSE(test_error)


    fig, axs = plt.subplots(2, 2)

    axs[0, 0].plot(input_data[0:numOfTrain, m-1],'r--',output_train,'b')
    axs[0, 0].set_title('Train')
    axs[0, 1].plot(input_data[numOfTrain:, m-1], 'g--', output_test,'b')
    axs[0, 1].set_title('Test')
    axs[1, 0].plot(mse_train, 'tab:orange')
    axs[1, 0].set_title('MSE Train')
    axs[1, 1].plot(mse_test, 'tab:green')
    axs[1, 1].set_title('MSE Test')




    # plt.subplot(411)
    # plt.plot(input_data[0:numOfTrain, 3],'r--',output_train,'b')
    # plt.title('Train')
    #
    # plt.subplot(412)
    # plt.plot(input_data[numOfTrain:, 3], 'g--', output_test,'b')
    # plt.title('Test')
    #
    # plt.subplot(421)
    # plt.plot(mse_train)
    # plt.subplot(422)
    # plt.plot(mse_test)

    plt.pause(0.01)
# endregion


plt.show()



# endregion

