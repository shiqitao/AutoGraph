class Param:

    def __init__(self, model, param, retry=2):
        self.__model = model
        self.__param = param
        self.__running = False
        self.__index = None
        self.__time_budget = None
        self.__result = None
        self.__retry = retry

    @property
    def model(self):
        return self.__model

    @property
    def param(self):
        return self.__param

    @property
    def running(self):
        return self.__running

    @running.setter
    def running(self, running):
        self.__running = running

    @property
    def index(self):
        return self.__index

    @index.setter
    def index(self, index):
        self.__index = index

    @property
    def time_budget(self):
        return self.__time_budget

    @time_budget.setter
    def time_budget(self, time_budget):
        self.__time_budget = time_budget

    @property
    def result(self):
        return self.__result

    @result.setter
    def result(self, result):
        self.__result = result

    @property
    def retry(self):
        return self.__retry

    @retry.setter
    def retry(self, retry):
        self.__retry = retry
