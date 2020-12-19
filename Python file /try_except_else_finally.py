try:
    A=int(input("Insert the integer:"))
    
    
    
    print("the radius of circle:",A)
    print("the circumstance of circle:",2*3.14*A)
    print("the area of circle:",3.14*A*A)
except:
    print("You don't insert the integer. Check it please!")  
    
else:
    
    print("the program operates well")
finally:
    print("Anyway, end the process whether it is true or not")