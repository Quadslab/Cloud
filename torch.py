import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

class NET(nn.Module):
  def __init__(self):
    super(NET, self).__init__()
  	self.lin1 = nn.Linear(10, 10)
    self.lin2 = nn.Linear(10, 10)
  
  def forward(self, x):
    x = F.relu(self.lin1(x))
    x = self.lin2(x)
    return x
  
  def num_flat_features(self, x):
    size = x.size()[1:]
    num = 1
    for i in size:
      num *= i
    return num
  
neuralnet = NET()

for i in range(100):
	input = Variable(torch.randn(10,10))

	out = neuralnet(input)
	
	x = [0,1,0,1,1,1,0,0,1,1]
	target = Variable(torch.Tensor([x for i in range(10)]))
	
	crit = nn.MSELoss()
	loss = crit(out, target)
	print(loss)
	
	neuralnet .zero_grad()
	loss.backward()
	optimizer = optim.SGD(neuralnet.parameters(), lr=0.01)
	optimizer.step()