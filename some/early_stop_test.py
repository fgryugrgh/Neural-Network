validation = [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]

def earlystop(x):
    counter = 0
    for i in range(len(x)):
        if counter == 10:
            print("stopped")
            return
        else:
            counter = counter + 1 if x[i] == x[i-1] else 0
    return counter

earlystop(validation)
