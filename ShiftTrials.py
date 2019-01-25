class Shift:
    def __init__(self, max_shift):
        self.max_shift = max_shift

    def shift(self, arr, t):
        if t > self.max_shift:
            raise Exception('Cant shift that hard')
        if t == self.max_shift:
            return arr[t:]  # Can't index by -0 :(
        return arr[t:-(self.max_shift - t)]
