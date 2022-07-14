import pickle as pkl

if __name__ == '__main__':
    with open('inputs.pkl', 'rb') as f:
        file = pkl.load(f)
        print(file['input_arrays'][0])
    