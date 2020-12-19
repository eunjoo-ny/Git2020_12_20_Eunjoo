def test():
    print("pass the A location")
    yield 1
    print("pass the B point")
    yield 2
    print("pass the C point")
    
outcome=test()

print("pass the D point")
a=next(outcome)
print(a)
print()
print("pass the E point")
b=next(outcome)
print(b)
print()
print("pass the F point")
c=next(outcome)
print(c)

# I don't know why this program doesn't operate

