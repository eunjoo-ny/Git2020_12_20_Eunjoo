def sum_all(start,end):
    output=0
    for i in range(start,end+1):
        output+=i
        return output
    print("from 0 to 100:",sum_all(0,100))
    print("from 0 to 1000:",sum_all(0,1000))
    print("from 50 to 100:",sum_all(50,100))
    print("from 500 to 1000:",sum_all(500,1000))
    
def total_sum(start=0,end=100,step=1):
    result=0
    for i in range(start,end+1,step):
        result+=i
        return result
    print("A.",total_sum(0,100,10))
    print("B.",total_sum(end=100))
    print("C.",total_sum(end=100,step=2))
    