def test():
    print("call the function")
    yield test
print("The pass of A")
test()
print("The pass of B")
test()
print(test())
