class AttentionLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(AttentionLayer, self).__init__()
        self.input_proj = nn.Linear(input_size, output_size, bias=False)
        self.output_proj = nn.Linear(output_size, output_size, bias=False)
