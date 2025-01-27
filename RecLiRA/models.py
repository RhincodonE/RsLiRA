import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuMF(nn.Module):
    def __init__(self, args, num_users, num_items):
        super(NeuMF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.factor_num_mf = args.factor_num
        self.factor_num_mlp = int(args.layers[0] // 2)
        self.layers = args.layers
        self.dropout = args.dropout

        # Embeddings for MLP part
        self.embedding_user_mlp = nn.Embedding(
            num_embeddings=self.num_users,
            embedding_dim=self.factor_num_mlp
        )
        self.embedding_item_mlp = nn.Embedding(
            num_embeddings=self.num_items,
            embedding_dim=self.factor_num_mlp
        )

        # Embeddings for MF part
        self.embedding_user_mf = nn.Embedding(
            num_embeddings=self.num_users,
            embedding_dim=self.factor_num_mf
        )
        self.embedding_item_mf = nn.Embedding(
            num_embeddings=self.num_items,
            embedding_dim=self.factor_num_mf
        )

        # MLP layers
        self.fc_layers = nn.ModuleList()
        for in_size, out_size in zip(self.layers[:-1], self.layers[1:]):
            self.fc_layers.append(nn.Linear(in_size, out_size))
            self.fc_layers.append(nn.ReLU())

        # Final linear layer combining MLP + MF
        self.affine_output = nn.Linear(
            in_features=self.layers[-1] + self.factor_num_mf,
            out_features=1
        )

        # Sigmoid activation for final prediction
        self.logistic = nn.Sigmoid()

        # Initialize weights
        self.init_weight()

    def init_weight(self):
        """Initialize embeddings and linear layers with common initialization strategies."""
        nn.init.normal_(self.embedding_user_mlp.weight, std=0.01)
        nn.init.normal_(self.embedding_item_mlp.weight, std=0.01)
        nn.init.normal_(self.embedding_user_mf.weight, std=0.01)
        nn.init.normal_(self.embedding_item_mf.weight, std=0.01)

        for module in self.fc_layers:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

        nn.init.xavier_uniform_(self.affine_output.weight)

        for module in self.modules():
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

    def forward(self, user_indices, item_indices):
        # MLP embeddings
        user_embedding_mlp = self.embedding_user_mlp(user_indices)
        item_embedding_mlp = self.embedding_item_mlp(item_indices)

        # MF embeddings
        user_embedding_mf = self.embedding_user_mf(user_indices)
        item_embedding_mf = self.embedding_item_mf(item_indices)

        # Concatenate user/item embeddings for MLP
        mlp_vector = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)

        # Element-wise multiplication for MF
        mf_vector = torch.mul(user_embedding_mf, item_embedding_mf)

        # Pass through MLP layers
        for layer in self.fc_layers:
            mlp_vector = layer(mlp_vector)

        # Concatenate MLP and MF parts
        vector = torch.cat([mlp_vector, mf_vector], dim=-1)

        # Final linear layer + Sigmoid
        logits = self.affine_output(vector)
        rating = self.logistic(logits)
        return rating.squeeze()
