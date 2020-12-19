pi=3.14
A=input("Insert the integer>")
if A.isdigit():
    B=int(A)
    print("the radius of circle:",B)
    print("the circumstance of circle:",2*pi*B)
    print("the area of circle:",pi*B*B)
else:
    print("You don't the integer. Pay attention to ^^")