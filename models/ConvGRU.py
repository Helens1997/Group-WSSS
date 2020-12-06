import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import init
class ConvGRUCell(nn.Module):
    
    def __init__(self, input_size, hidden_size,  kernel_size):
        super(ConvGRUCell,self).__init__()
        self.input_size  = input_size
        self.cuda_flag   = True
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.padding = int((kernel_size - 1) / 2)
        self.reset_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=self.padding)
        self.update_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=self.padding)
        self.out_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=self.padding)

    def forward(self, input_, prev_state):

        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            if torch.cuda.is_available():
                prev_state = Variable(torch.zeros(state_size)).cuda()
            else:
                prev_state = Variable(torch.zeros(state_size))

        stacked_inputs = torch.cat([input_, prev_state], dim=1)
        update = F.sigmoid(self.update_gate(stacked_inputs))
        reset = F.sigmoid(self.reset_gate(stacked_inputs))
        out_inputs = F.tanh(self.out_gate(torch.cat([input_, prev_state * reset], dim=1)))
        new_state = prev_state * (1 - update) + out_inputs * update

        return new_state
