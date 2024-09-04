import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, dim, batch_norm=True, dropout=0.2):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

        self.batch_norm = batch_norm
        self.bn1 = nn.BatchNorm1d(dim) if batch_norm else None
        self.bn2 = nn.BatchNorm1d(dim) if batch_norm else None

        self.dropout = nn.Dropout(dropout) if dropout else None

        # Apply Glorot initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        residual = x
        out = self.fc1(x)

        if self.batch_norm:
            out = self.bn1(out)
        out = F.relu(out)

        if self.dropout:
            out = self.dropout(out)

        out = self.fc2(out)
        if self.batch_norm:
            out = self.bn2(out)

        out += residual
        out = F.relu(out)

        return out


class ResnetModel(nn.Module):
    def __init__(self, state_dim, one_hot_depth=6, h1_dim=256, resnet_dim=256, num_resnet_blocks=4,
                 policy_out_dim=12, value_out_dim=1, batch_norm=True, dropout=0.2):
        super(ResnetModel, self).__init__()
        self.one_hot_depth = one_hot_depth
        self.state_dim = state_dim
        self.num_resnet_blocks = num_resnet_blocks
        self.batch_norm = batch_norm

        # Shared layers
        if one_hot_depth > 0:
            self.fc1 = nn.Linear(self.state_dim * self.one_hot_depth, h1_dim)
        else:
            self.fc1 = nn.Linear(self.state_dim, h1_dim)
        self.bn1 = nn.BatchNorm1d(h1_dim) if batch_norm else None
        self.dropout1 = nn.Dropout(dropout) if dropout else None

        self.fc2 = nn.Linear(h1_dim, resnet_dim)
        self.bn2 = nn.BatchNorm1d(resnet_dim) if batch_norm else None
        self.dropout2 = nn.Dropout(dropout) if dropout else None

        # Residual Blocks (Shared)
        self.blocks = nn.ModuleList([ResidualBlock(resnet_dim, batch_norm, dropout)
                                     for _ in range(self.num_resnet_blocks)])

        # Policy Network (outputs move probabilities)
        self.policy_fc1 = nn.Linear(resnet_dim, resnet_dim)
        self.policy_bn1 = nn.BatchNorm1d(resnet_dim) if batch_norm else None
        self.policy_fc_out = nn.Linear(resnet_dim, policy_out_dim)

        # Value Network (outputs a single value)
        self.value_fc1 = nn.Linear(resnet_dim, resnet_dim)
        self.value_bn1 = nn.BatchNorm1d(resnet_dim) if batch_norm else None
        self.value_fc_out = nn.Linear(resnet_dim, value_out_dim)

        # Apply Glorot initialization to all layers
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # Preprocess input
        if self.one_hot_depth > 0:
            x = F.one_hot(x.long(), self.one_hot_depth)
            x = x.float()
            x = x.view(-1, self.state_dim * self.one_hot_depth)
        else:
            x = x.float()

        # Shared layers
        x = self.fc1(x)
        if self.batch_norm:
            x = self.bn1(x)
        x = F.relu(x)
        if self.dropout1:
            x = self.dropout1(x)

        x = self.fc2(x)
        if self.batch_norm:
            x = self.bn2(x)
        x = F.relu(x)
        if self.dropout2:
            x = self.dropout2(x)

        # Residual Blocks (Shared)
        for block in self.blocks:
            x = block(x)

        # Policy Network Branch
        policy_x = self.policy_fc1(x)
        if self.batch_norm:
            policy_x = self.policy_bn1(policy_x)
        policy_x = F.relu(policy_x)
        policy_out = self.policy_fc_out(policy_x)
        policy_out = F.softmax(policy_out, dim=-1)  # Apply softmax to get probabilities

        # Value Network Branch
        value_x = self.value_fc1(x)
        if self.batch_norm:
            value_x = self.value_bn1(value_x)
        value_x = F.relu(value_x)
        value_out = self.value_fc_out(value_x)

        return [policy_out, value_out]


"""
    # Example usage:
    # Initialize the ResnetModel with specified parameters
    # model = ResnetModel(
    #     state_dim=54,            # Size of the Rubik's cube state (54 stickers)
    #     one_hot_depth=6,         # Number of unique colors (6 for Rubik's cube)
    #     h1_dim=256,              # Dimension of the first hidden layer
    #     resnet_dim=256,          # Dimension of the residual block layers
    #     num_resnet_blocks=4,     # Number of residual blocks
    #     policy_out_dim=12,       # Number of possible actions (12 move probabilities)
    #     value_out_dim=1,         # Single value output
    #     batch_norm=True,         # Whether to use batch normalization
    #     dropout=0.3              # Dropout rate
    # )

    To perform a forward pass:
    input_tensor = torch.randn((batch_size, state_dim))  # Replace batch_size with the desired batch size
    policy_output, value_output = model(input_tensor)
"""
