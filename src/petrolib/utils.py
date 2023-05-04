class MnemonicError(Exception):
    '''
    Exception raised if the log curve/menemonic passed is not found
    '''
    def __init__(self, msg):
        self.msg = msg
        super().__init__(self.msg)