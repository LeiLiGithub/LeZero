# This file is used to test android develop environment
is_android = True
# is_android = False

def sum(a, b):
    print("You are calling sum()...")
    sum = a+b
    print (a, "+", b, "=", sum)
    return sum

if not is_android:
    sum(100, 200)