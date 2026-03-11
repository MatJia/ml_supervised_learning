import random as rd

def generate_test_data() -> tuple[int, int]:

    k = round(rd.uniform(-3,3), 2)
    b = rd.randint(-5000,5000)
    with open("../data/linear_regression.csv","w") as f:
        f.write("x,y\n")
        for i in range(0,10000):
            this_x = rd.randint(-100000,100000)
            this_y = round(k * this_x + b, 3)
            f.write(f"{this_x},{this_y}\n")
    return k, b