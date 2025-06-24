import numpy as np

def fold(k: int, index: int):
    trains = []
    tests = []
    with open("../dataset/train.npy", "rb") as fp:
        for count in range(10):
            train = np.load(fp)
            train[:, 1:785][train[:, 1:785] > 0] = 1
            tests.append(train[index*k:(index+1)*k])
            trains.append(np.delete(train, range(index*k,(index+1)*k), axis = 0))
    weights = []
    for i in range(10):
        weights.append(np.sum(trains[i][:, 1:785], axis=0) / len(trains[i][:, 1:785]))
    fin = []
    for index in range(10):
        res = [0]*10
        for value in tests[index][:, 1:785] == 0:
            temp = np.array(weights).copy()
            temp[:, value] = 1 - temp[:, value]
            res[np.argmax(np.prod(temp, axis=1))] += 1
        fin.append(res)
    return fin

def kFold(k):
    sol = []
    for i in range(int(5000/k)):
        sol.append(np.sum(fold(k,i)*np.identity(10))/(k/10))
    return np.array(sol)

print(kFold(20))


