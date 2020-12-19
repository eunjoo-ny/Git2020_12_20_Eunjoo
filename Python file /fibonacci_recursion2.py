counter=0
def fibonacci(n):
    print("Find the fibonacci({}):".format(n))
    global counter
    counter+=1
    
    if n==1:
        return 1
    elif n==2:
        return 1
    else:
        return fibonacci(n-1)+fibonacci(n-2)
fibonacci(10)
print()
print("the number of addition applying computation in fibonacci(10) is {}".format(counter))