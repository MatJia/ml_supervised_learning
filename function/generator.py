def generate_test_data():
    import random as rd
    k = round(rd.uniform(0,3), 2)
    b = rd.randint(-5000,5000)
    with open("../data/linear_regression.csv","w") as f:
        f.write("x,y\n")
        for i in range(0,10000):
            this_x = rd.randint(-100000,100000)
            this_y = k * this_x + b
            f.write(f"{this_x},{this_y}\n")