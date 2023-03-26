class Detection:
    def __init__(self, id, label, xy = (0,0)):
        self.id = id
        self.name = label
        self.xy = xy


    def printme(self):
        print(f"{self.name} {self.conf} {self.x} {self.y}")
    
    def get_tuple(self):
        return (self.x, self.y)
