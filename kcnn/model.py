import torch.nn as nn
import torch.nn.functional as F
from init import xavier_normal,xavier_uniform,orthogonal

# CNN Model
class CNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dim, num_kernels, max_document_length):
        super(CNN, self).__init__()
        self.max_document_length = max_document_length
        self.conv = nn.Conv3d(1, input_size, (1, 1, dim), padding=0)
        self.fc1 = nn.Linear(input_size*num_kernels, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.init_weights()

    def init_weights(self):
        xavier_uniform(self.conv.weight.data)
        xavier_normal(self.fc1.weight.data)
        xavier_normal(self.fc2.weight.data)

    def forward(self, x_in):
        out = F.relu(F.max_pool3d(self.conv(x_in), (1, self.max_document_length,1)))
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.dropout(out, training=self.training)
        out = self.fc2(out)
        return F.log_softmax(out)
           