import datasets as ds


mnist_train = ds.mnist(True)
mnist_validation = ds.mnist(False)

mnist_train.__getitem__(0)