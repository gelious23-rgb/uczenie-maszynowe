from Model import Model

if __name__ == '__main__':
    m = Model()
    m.load_trained(path=r'trained_models\my_model3.h5')
    print('Starting test')
    accuracy = m.test()
    print('Test was finished')
    print(f'Accuracy is {accuracy}%')
