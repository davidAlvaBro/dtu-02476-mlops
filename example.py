import torch as th
from tqdm import tqdm


if __name__ == '__main__':
    print(th.cuda.is_available())
    print(th.arange(3))
    
    for i in tqdm(range(5)):
        print(i)