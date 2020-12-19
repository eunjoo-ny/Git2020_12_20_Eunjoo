def consecutive_addition(n):
    if n==0:
        return 0
    else:
        return n+consecutive_addition(n-1)
    
print("the sum of from 1 to 100:",consecutive_addition(100))
#print("the sum of from 1 to 1500:",consecutive_addition(1500))
