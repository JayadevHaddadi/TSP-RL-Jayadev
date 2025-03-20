class GraphPointerNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embedder = GCNConv(2, args.hidden_dim)  # Process coordinates as graph
        self.lstm = nn.LSTM(args.hidden_dim, args.hidden_dim, batch_first=True)
        self.pointer = AttentionPointer(args.hidden_dim)
