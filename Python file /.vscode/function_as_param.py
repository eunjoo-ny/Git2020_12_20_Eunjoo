def times(func):
    for i in range(5):
        func()
def hello() : 
    print("Hello my friends")  
#def hello(n):
    #a=int(n**4+n+3)
    #print(a) .....I 'm curious about in this case. 
    #times(hello(10))    
times(hello)