def call_10_times(func):
    for i in range(10):
        func(i)
def print_hello(number):
    print("안녕하세요",number)
call_10_times(print_hello)

def call_10_times(func):
    for i in range(10):
        #콜백 함수:위 함수의 과정을 줄여줌
        func(i)
call_10_times(lambda number:print("안녕하세요",number))