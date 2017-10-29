import torch 
import torch.nn as nn
import torchvision.transforms as transforms
import torch.utils.data as utils
from torch.autograd import Variable
import numpy as np
from utils import compute_nystrom,create_train_test_loaders
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from model import CNN

# Parameters
data_file = "IMDB-BINARY"
use_node_labels = True
unlabeled_data_files = ["IMDB-BINARY", "IMDB-MULTI", "REDDIT-BINARY", "REDDIT-MULTI-5K", "COLLAB", "SYNTHETIC"]
if data_file in unlabeled_data_files:
    use_node_labels = False
community_detection ="louvain"

# Hyper Parameters
dim = 200
batch_size = 64
num_epochs = 100
num_filters = 256
hidden_size = 128
learning_rate = 0.0001

if use_node_labels:
   from graph_kernels_labeled import sp_kernel, wl_kernel
else:
   from graph_kernels import sp_kernel, wl_kernel

# Choose kernels
kernels=[wl_kernel]
num_kernels = len(kernels)

print("Computing feature maps...")
Q, subgraphs, labels,shapes = compute_nystrom(data_file, use_node_labels, dim, community_detection, kernels)

M=np.zeros((shapes[0],shapes[1],len(kernels)))
for idx,k in enumerate(kernels):
    M[:,:,idx]=Q[idx]

Q=M

# Binarize labels
le = LabelEncoder()
y = le.fit_transform(labels)

# Build vocabulary
max_document_length = max([len(x.split(" ")) for x in subgraphs])
x = np.zeros((len(subgraphs), max_document_length), dtype=np.int32)
for i in range(len(subgraphs)):
	communities = subgraphs[i].split()
	for j in range(len(communities)):
		x[i,j] = int(communities[j])


kf = KFold(n_splits=10, random_state=None)
kf.shuffle=True
accs=[];
it = 0

print("Starting cross-validation...")

for train_index, test_index in kf.split(x):
	it += 1
	x_train, x_test = x[train_index], x[test_index]
	y_train, y_test = y[train_index], y[test_index]

	train_loader, test_loader = create_train_test_loaders(Q, x_train, x_test, y_train, y_test, batch_size)
		   
	cnn = CNN(input_size=num_filters, hidden_size=hidden_size, num_classes=np.unique(y).size, dim=dim, num_kernels=num_kernels, max_document_length=max_document_length)
	if torch.cuda.is_available():
		cnn.cuda()

	# Loss and Optimizer
	if torch.cuda.is_available():
		criterion = nn.CrossEntropyLoss().cuda()
	else:
		criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

	# Train the Model
	for epoch in range(num_epochs):
		for i, (graphs, labels) in enumerate(train_loader):
		    graphs = Variable(graphs)
		    labels = Variable(labels)
		    
		    optimizer.zero_grad()
		    outputs = cnn(graphs)
		    if torch.cuda.is_available():
		    	loss = criterion(outputs, labels.cuda())
		    else:
		    	loss = criterion(outputs, labels)
		    loss.backward()
		    optimizer.step()
		    #if (i+1) % 10 == 0:
		    #    print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' %(epoch+1, num_epochs, i+1, x.shape[0]//64, loss.data[0]))

	# Test the Model
	cnn.eval() 
	correct = 0
	total = 0
	for graphs, labels in test_loader:
		graphs = Variable(graphs)
		outputs = cnn(graphs)
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		if torch.cuda.is_available():
			correct += (predicted == labels.cuda()).sum()
		else:
			correct += (predicted == labels).sum()

	acc = (100 * correct / total)
	accs.append(acc)
	print("Accuracy at iteration "+ str(it) +": " + str(acc))
    
print("Average accuracy: ", np.mean(accs))