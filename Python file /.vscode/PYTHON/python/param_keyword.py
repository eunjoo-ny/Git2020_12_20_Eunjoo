def positive_view(*values,n=2):
    for i in range(n):
        for value in values:
            print(value)
        print()
positive_view("I want to","eat","delicious","food",n=3)