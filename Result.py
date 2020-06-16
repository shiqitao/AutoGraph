class Result:

    def __init__(self, result, loss_train, loss_valid, acc_train, acc_valid, epoch):
        self.__result = result
        self.__loss_train = loss_train
        self.__loss_valid = loss_valid
        self.__acc_train = acc_train
        self.__acc_valid = acc_valid
        self.__epoch = epoch

    @property
    def result(self):
        return self.__result

    @property
    def loss_train(self):
        return self.__loss_train

    @property
    def loss_valid(self):
        return self.__loss_valid

    @property
    def acc_train(self):
        return self.__acc_train

    @property
    def acc_valid(self):
        return self.__acc_valid

    @property
    def epoch(self):
        return self.__epoch
