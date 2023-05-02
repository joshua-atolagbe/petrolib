class MnemonicError(Exception):
    '''
    Exception raised if the log curve/menemonic passed is not found
    '''
    def __init__(self, curve, msg):
        self.curve = curve
        self.msg = msg
        super().__init__(self.msg)