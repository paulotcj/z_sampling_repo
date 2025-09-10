
class CustomClass():
    def __init__(self):
        pass

    def __call__(self, printparam, *args):
        self.custom_method(printparam, *args)

    def custom_method(self, printparam, *args):
        print(printparam)

class CustomClass2(CustomClass):
    def __init__(self):
        pass

class CustomClass3(CustomClass):
    def __init__(self):
        pass

    def custom_method(self, printparam, param2):
        print(f'CustomClass3 - {printparam} - param2: {param2}')




obj = CustomClass()
str = 'hello'
obj(str)
print('------------')
obj2 = CustomClass2()
obj2(str)
print('------------')
obj3 = CustomClass3()
obj3(str, 12)
class CustomClass():
    def __init__(self):
        pass

    def __call__(self, printparam, *args):
        self.custom_method(printparam, *args)

    def custom_method(self, printparam, *args):
        pass