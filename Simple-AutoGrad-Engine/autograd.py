class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.grad = 0.0
        self._backward = lambda : None

    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad * 1
            other.grad += out.grad * 1
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
        out._backward = _backward
        return out

    def __sub__(self, other):
        out = Value(self.data - other.data, (self, other), "-")
        def _backward():
            self.grad += out.grad * 1
            other.grad += out.grad * -1
        out._backward = _backward
        return out
    def __truediv__(self, other):
        out = Value(self.data / other.data, (self, other), "/")
        def _backward():
            self.grad += out.grad * (1 / other.data)
            other.grad += out.grad * -(self.data / (other.data**2) )
        out._backward = _backward
        return out

    
    def backward(self):
        topo = []
        visisted = set()

        def buildTopo(v):
            if v not in visisted:
                visisted.add(v)

                for child in v._prev:
                    buildTopo(child)
    
                topo.append(v)
        buildTopo(self)
        self.grad = 1.0

        for node in reversed(topo):
            node._backward()
    

a = Value(2.0)
b = Value(-3.0)
c = a / b
c.backward()

print(a.grad)
print(b.grad)