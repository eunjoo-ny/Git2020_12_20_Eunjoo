number=input("insert your favorite-integer number>")
last_character=number[-1]
last_number=int(last_character)
if last_number%2==0:
    print("Your number is even number")
if last_number%2==1:
    print("Your nubmer is odd number")