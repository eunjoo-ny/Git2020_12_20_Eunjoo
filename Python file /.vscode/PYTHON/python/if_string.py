output=int(input("Insert the integer>"))
if output%2==0 and output%3==0:
    print("Inserted number is {} and {} is 6-multiple".format(output,output))
elif output%2==0:
    print("Inserted number is {} and even".format(output))
elif output%3==0:
    print("Inserted number is {} and odd".format(output))
else:
    print("It is not both 2-multiple and 3-multiple")