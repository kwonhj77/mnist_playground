from mnist import MNIST

mndata = MNIST(r'C:\Users\Will Haley\Documents\GitHub\mnist_playground\mnist_dataset')

# path = r'C:\Users\Will Haley\Documents\GitHub\mnist_playground\mnist_dataset\t10k-images-idx3-ubyte\t10k-images-idx3-ubyte'

# file = open(path)

tr_images, tr_labels = mndata.load_training()
te_images, te_labels = mndata.load_testing()