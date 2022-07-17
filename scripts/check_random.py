import random

for n in range(10):
    random.seed(1)
    random_indices = random.sample(range(10), n)
    print(random_indices)

random.seed(1)
print(random.sample(range(10), 3))
random.seed(1)
print(random.sample(range(10), 6))
