import numpy as np
import pandas as pd

#init commit
def main():
    data = pd.read_csv('./data/movie_data.csv')
    print(data)

if __name__ == "__main__":
    main()