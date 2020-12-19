
with open("basic.txt","w")as file:
    file.write("The weather seems to be depressed because there is sad new.")


with open("basic.txt","r") as file:
    content=file.read()
    
print(content)
