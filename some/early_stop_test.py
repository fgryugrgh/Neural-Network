validation = [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]

def earlystop(x):
    counter = 0
    for i in range(len(x)):
        if counter == 10:
            print("stopped")
            return True
        else:
            counter = counter + 1 if x[i] == x[i-1] else 0
            print("good")

for i in range (5):
    if earlystop(validation):
        break
    print(f"{i}. good")
    
