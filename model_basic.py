import torch
import torch.nn as nn
import torch.nn.functional as F 

class BPR(nn.Module):
    def __init__(self, user_num, item_num, factor_num, num_layers,
                 dropout, model, GMF_model=None, MLP_model=None):
        """
        Initialize the Bayesian Personalized Ranking (BPR) model.

        Args:
        - user_num (int): Number of users in the dataset.
        - item_num (int): Number of items in the dataset.
        - factor_num (int): Number of latent factors to use for user and item embeddings.
        - num_layers (int): Number of layers in the MLP part of the model.
        - dropout (float): Dropout probability for the MLP layers.
        - model (str): Type of model to use ('MLP', 'GMF', or 'NeuMF-pre').
        - GMF_model (nn.Module, optional): Pre-trained GMF model to initialize from.
        - MLP_model (nn.Module, optional): Pre-trained MLP model to initialize from.
        """
        super(BPR, self).__init__()
        self.dropout = dropout
        self.model = model
        
        # Embedding layers
        self.embed_user_MLP = nn.Embedding(user_num, factor_num)
        self.embed_item_MLP = nn.Embedding(item_num, factor_num)

        # MLP layers
        MLP_modules = []
        for i in range(num_layers):
            if i == 0:
                input_size = factor_num * 2
            else:
                input_size = factor_num
            MLP_modules.append(nn.Dropout(p=self.dropout))
            MLP_modules.append(nn.Linear(input_size, input_size // 2 if i == 0 else input_size))
            MLP_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*MLP_modules)

        # Prediction layer
        if self.model in ['MLP', 'GMF']:
            predict_size = factor_num 
        else:
            predict_size = factor_num * 2
        self.predict_layer = nn.Linear(predict_size, 1)

        # Weight initialization
        self._init_weight_()

    def _init_weight_(self):
        """
        Initialize weights for embeddings, MLP layers, and prediction layer.
        """
        nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
        nn.init.normal_(self.embed_item_MLP.weight, std=0.01)

        for m in self.MLP_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity='sigmoid')

        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    def forward_one_item(self, user, item):
        """
        Forward pass for a single user-item pair.

        Args:
        - user (Tensor): Tensor of user indices.
        - item (Tensor): Tensor of item indices.

        Returns:
        - prediction (Tensor): Predicted scores for the user-item pair.
        """
        embed_user_MLP = self.embed_user_MLP(user)
        embed_item_MLP = self.embed_item_MLP(item)
        
        interaction = torch.cat((embed_user_MLP, embed_item_MLP), -1)
        output_MLP = self.MLP_layers(interaction)

        if self.model == 'MLP':
            concat = output_MLP
        else:
            # output_GMF = embed_user_GMF * embed_item_GMF  # This is commented out in the original code
            # concat = torch.cat((output_GMF, output_MLP), -1)  # Here output_GMF is not defined
            concat = output_MLP

        prediction = self.predict_layer(concat)
        return prediction.view(-1)

    def forward(self, user, item_i, item_j):
        """
        Forward pass for a triplet of user and two items (positive and negative).

        Args:
        - user (Tensor): Tensor of user indices.
        - item_i (Tensor): Tensor of positive item indices.
        - item_j (Tensor): Tensor of negative item indices.

        Returns:
        - prediction_i (Tensor): Predicted scores for the user-positive item pair.
        - prediction_j (Tensor): Predicted scores for the user-negative item pair.
        """
        prediction_i = self.forward_one_item(user, item_i)
        prediction_j = self.forward_one_item(user, item_j)

        return prediction_i, prediction_j

class BPR_MACR(nn.Module):
    def __init__(self, user_num, item_num, factor_num, num_layers,
                 dropout, model, GMF_model=None, MLP_model=None):
        """
        Initialize the BPR-MACR model.

        Args:
        - user_num (int): Number of users in the dataset.
        - item_num (int): Number of items in the dataset.
        - factor_num (int): Number of latent factors to use for user and item embeddings.
        - num_layers (int): Number of layers in the MLP part of the model.
        - dropout (float): Dropout probability for the MLP layers.
        - model (str): Type of model to use ('MLP', 'GMF', or 'NeuMF-pre').
        - GMF_model (nn.Module, optional): Pre-trained GMF model to initialize from.
        - MLP_model (nn.Module, optional): Pre-trained MLP model to initialize from.
        """
        super(BPR_MACR, self).__init__()
        self.dropout = dropout
        self.model = model
        self.MLP_model = MLP_model
        
        self.embed_user_MLP = nn.Embedding(user_num, factor_num)
        self.embed_item_MLP = nn.Embedding(item_num, factor_num)

        # MACR (Macro Coefficients for Users and Items)
        self.macr_user = nn.Linear(factor_num * 2, 1)  # Assuming factor_num * 2 as input size
        self.macr_item = nn.Linear(factor_num * 2, 1)  # Assuming factor_num * 2 as input size

        MLP_modules = []
        for i in range(num_layers):
            if i == 0:
                input_size = factor_num * 2
                MLP_modules.append(nn.Dropout(p=self.dropout))
                MLP_modules.append(nn.Linear(input_size, input_size // 2))
                MLP_modules.append(nn.ReLU())
            else:
                input_size = factor_num
                MLP_modules.append(nn.Dropout(p=self.dropout))
                MLP_modules.append(nn.Linear(input_size, input_size))
                MLP_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*MLP_modules)

        if self.model in ['MLP', 'GMF']:
            predict_size = factor_num 
        else:
            predict_size = factor_num * 2
        self.predict_layer = nn.Linear(predict_size, 1)

        self._init_weight_()

    def _init_weight_(self):
        """
        Initialize weights for embeddings, MLP layers, MACR coefficients, and prediction layer.
        """
        nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
        nn.init.normal_(self.embed_item_MLP.weight, std=0.01)

        for m in self.MLP_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity='sigmoid')

        nn.init.xavier_uniform_(self.macr_user.weight)
        self.macr_user.bias.data.zero_()
        nn.init.xavier_uniform_(self.macr_item.weight)
        self.macr_item.bias.data.zero_()

    def forward_one_item(self, user, item):
        """
        Forward pass for a single user-item pair.

        Args:
        - user (Tensor): Tensor of user indices.
        - item (Tensor): Tensor of item indices.

        Returns:
        - prediction (Tensor): Predicted score for the user-item pair.
        """
        embed_user_MLP = self.embed_user_MLP(user)
        embed_item_MLP = self.embed_item_MLP(item)
        
        interaction = torch.cat((embed_user_MLP, embed_item_MLP), -1)
        output_MLP = self.MLP_layers(interaction)

        if self.model == 'MLP':
            concat = output_MLP
        else:
            # output_GMF = embed_user_GMF * embed_item_GMF  # This is commented out in the original code
            # concat = torch.cat((output_GMF, output_MLP), -1)  # Here output_GMF is not defined
            concat = output_MLP

        prediction = self.predict_layer(concat)
        return prediction.view(-1)

    def forward(self, user, item_i, item_j):
        """
        Forward pass for a triplet of user and two items (positive and negative).

        Args:
        - user (Tensor): Tensor of user indices.
        - item_i (Tensor): Tensor of positive item indices.
        - item_j (Tensor): Tensor of negative item indices.

        Returns:
        - prediction_i (Tensor): Predicted score for the user-positive item pair.
        - prediction_j (Tensor): Predicted score for the user-negative item pair.
        """
        prediction_i = self.forward_one_item(user, item_i)
        prediction_j = self.forward_one_item(user, item_j)

        return prediction_i, prediction_j
		
class NCF(nn.Module):
    def __init__(self, user_num, item_num, factor_num, num_layers,
                 dropout, model, GMF_model=None, MLP_model=None):
        """
        Initialize the Neural Collaborative Filtering (NCF) model.

        Args:
        - user_num (int): Number of users in the dataset.
        - item_num (int): Number of items in the dataset.
        - factor_num (int): Number of latent factors to use for user and item embeddings.
        - num_layers (int): Number of layers in the MLP part of the model.
        - dropout (float): Dropout probability for the MLP layers.
        - model (str): Type of model to use ('MLP', 'GMF', or 'NeuMF-pre').
        - GMF_model (nn.Module, optional): Pre-trained GMF model to initialize from.
        - MLP_model (nn.Module, optional): Pre-trained MLP model to initialize from.
        """
        super(NCF, self).__init__()
        self.dropout = dropout
        self.model = model
        self.GMF_model = GMF_model
        self.MLP_model = MLP_model
        
        # Embedding layers for GMF and MLP
        self.embed_user_GMF = nn.Embedding(user_num, factor_num)
        self.embed_item_GMF = nn.Embedding(item_num, factor_num)
        self.embed_user_MLP = nn.Embedding(user_num, factor_num)
        self.embed_item_MLP = nn.Embedding(item_num, factor_num)

        # MLP layers
        MLP_modules = []
        for i in range(num_layers):
            if i == 0:
                input_size = factor_num * 2
                MLP_modules.append(nn.Dropout(p=self.dropout))
                MLP_modules.append(nn.Linear(input_size, input_size // 2))
                MLP_modules.append(nn.ReLU())
            else:
                input_size = factor_num
                MLP_modules.append(nn.Dropout(p=self.dropout))
                MLP_modules.append(nn.Linear(input_size, input_size))
                MLP_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*MLP_modules)

        # Prediction layer
        if self.model in ['MLP', 'GMF']:
            predict_size = factor_num 
        else:
            predict_size = factor_num * 2
        self.predict_layer = nn.Linear(predict_size, 1)

        # Initialize weights
        self._init_weight_()

    def _init_weight_(self):
        """
        Initialize weights for embeddings, MLP layers, and prediction layer.
        """
        if not self.model == 'NeuMF-pre':
            nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
            nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
            nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
            nn.init.normal_(self.embed_item_MLP.weight, std=0.01)

            for m in self.MLP_layers:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
            nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity='sigmoid')

            for m in self.modules():
                if isinstance(m, nn.Linear) and m.bias is not None:
                    m.bias.data.zero_()

    def forward_one_item(self, user, item):
        """
        Forward pass for predicting score for a single user-item pair.

        Args:
        - user (Tensor): Tensor of user indices.
        - item (Tensor): Tensor of item indices.

        Returns:
        - prediction (Tensor): Predicted score for the user-item pair.
        """
        if not self.model == 'MLP':
            embed_user_GMF = self.embed_user_GMF(user)
            embed_item_GMF = self.embed_item_GMF(item)
            output_GMF = embed_user_GMF * embed_item_GMF

        if not self.model == 'GMF':
            embed_user_MLP = self.embed_user_MLP(user)
            embed_item_MLP = self.embed_item_MLP(item)
            interaction = torch.cat((embed_user_MLP, embed_item_MLP), -1)
            output_MLP = self.MLP_layers(interaction)

        if self.model == 'GMF':
            concat = output_GMF
        elif self.model == 'MLP':
            concat = output_MLP
        else:  # NeuMF
            concat = torch.cat((output_GMF, output_MLP), -1)

        prediction = self.predict_layer(concat)
        return prediction.view(-1)

    def forward(self, user, item_i, item_j):
        """
        Forward pass for a triplet of user and two items (positive and negative).

        Args:
        - user (Tensor): Tensor of user indices.
        - item_i (Tensor): Tensor of positive item indices.
        - item_j (Tensor): Tensor of negative item indices.

        Returns:
        - prediction_i (Tensor): Predicted score for the user-positive item pair.
        - prediction_j (Tensor): Predicted score for the user-negative item pair.
        """
        prediction_i = self.forward_one_item(user, item_i)
        prediction_j = self.forward_one_item(user, item_j)

        return prediction_i, prediction_j

class NCF_MACR(nn.Module):
    def __init__(self, user_num, item_num, factor_num, num_layers,
                 dropout, model, GMF_model=None, MLP_model=None):
        """
        Initialize the Neural Collaborative Filtering model with MACR coefficients.

        Args:
        - user_num (int): Number of users in the dataset.
        - item_num (int): Number of items in the dataset.
        - factor_num (int): Number of latent factors to use for user and item embeddings.
        - num_layers (int): Number of layers in the MLP part of the model.
        - dropout (float): Dropout probability for the MLP layers.
        - model (str): Type of model to use ('MLP', 'GMF', or 'NeuMF-pre').
        - GMF_model (nn.Module, optional): Pre-trained GMF model to initialize from.
        - MLP_model (nn.Module, optional): Pre-trained MLP model to initialize from.
        """
        super(NCF_MACR, self).__init__()
        self.dropout = dropout
        self.model = model
        self.GMF_model = GMF_model
        self.MLP_model = MLP_model
        
        # Embedding layers for GMF and MLP
        self.embed_user_GMF = nn.Embedding(user_num, factor_num)
        self.embed_item_GMF = nn.Embedding(item_num, factor_num)
        self.embed_user_MLP = nn.Embedding(user_num, factor_num)
        self.embed_item_MLP = nn.Embedding(item_num, factor_num)

        # MACR coefficient embeddings
        self.macr_user = nn.Embedding(user_num, 1)
        self.macr_item = nn.Embedding(item_num, 1)

        # MLP layers
        MLP_modules = []
        for i in range(num_layers):
            if i == 0:
                input_size = factor_num * 2
                MLP_modules.append(nn.Dropout(p=self.dropout))
                MLP_modules.append(nn.Linear(input_size, input_size // 2))
                MLP_modules.append(nn.ReLU())
            else:
                input_size = factor_num
                MLP_modules.append(nn.Dropout(p=self.dropout))
                MLP_modules.append(nn.Linear(input_size, input_size))
                MLP_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*MLP_modules)

        # Prediction layer
        if self.model in ['MLP', 'GMF']:
            predict_size = factor_num 
        else:
            predict_size = factor_num * 2
        self.predict_layer = nn.Linear(predict_size, 1)

        # Initialize weights
        self._init_weight_()

    def _init_weight_(self):
        """
        Initialize weights for embeddings, MLP layers, and prediction layer.
        """
        if not self.model == 'NeuMF-pre':
            nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
            nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
            nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
            nn.init.normal_(self.embed_item_MLP.weight, std=0.01)
            nn.init.normal_(self.macr_user.weight, std=0.01)
            nn.init.normal_(self.macr_item.weight, std=0.01)

            for m in self.MLP_layers:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
            nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity='sigmoid')

            for m in self.modules():
                if isinstance(m, nn.Linear) and m.bias is not None:
                    m.bias.data.zero_()

    def forward_one_item(self, user, item):
        """
        Forward pass for predicting score for a single user-item pair.

        Args:
        - user (Tensor): Tensor of user indices.
        - item (Tensor): Tensor of item indices.

        Returns:
        - prediction (Tensor): Predicted score for the user-item pair.
        """
        if not self.model == 'MLP':
            embed_user_GMF = self.embed_user_GMF(user)
            embed_item_GMF = self.embed_item_GMF(item)
            output_GMF = embed_user_GMF * embed_item_GMF

        if not self.model == 'GMF':
            embed_user_MLP = self.embed_user_MLP(user)
            embed_item_MLP = self.embed_item_MLP(item)
            interaction = torch.cat((embed_user_MLP, embed_item_MLP), -1)
            output_MLP = self.MLP_layers(interaction)

        if self.model == 'GMF':
            concat = output_GMF
        elif self.model == 'MLP':
            concat = output_MLP
        else:  # NeuMF
            concat = torch.cat((output_GMF, output_MLP), -1)

        prediction = self.predict_layer(concat)
        return prediction.view(-1)

    def forward(self, user, item_i, item_j):
        """
        Forward pass for a triplet of user and two items (positive and negative).

        Args:
        - user (Tensor): Tensor of user indices.
        - item_i (Tensor): Tensor of positive item indices.
        - item_j (Tensor): Tensor of negative item indices.

        Returns:
        - prediction_i (Tensor): Predicted score for the user-positive item pair.
        - prediction_j (Tensor): Predicted score for the user-negative item pair.
        """
        prediction_i = self.forward_one_item(user, item_i)
        prediction_j = self.forward_one_item(user, item_j)

        return prediction_i, prediction_j

class MF_BPR(nn.Module):
    def __init__(self, user_num, item_num, factor_num, num_layers,
                 dropout, model, GMF_model=None, MLP_model=None):
        """
        Initialize Matrix Factorization with Bayesian Personalized Ranking (MF-BPR) model.

        Args:
        - user_num (int): Number of users in the dataset.
        - item_num (int): Number of items in the dataset.
        - factor_num (int): Number of latent factors to use for user and item embeddings.
        - num_layers (int): Number of layers in the MLP part of the model (not used in MF-BPR).
        - dropout (float): Dropout probability (not used in MF-BPR).
        - model (str): Type of model to use ('MLP', 'GMF', or others, but MF-BPR only supports 'MLP').
        - GMF_model (nn.Module, optional): Pre-trained GMF model (not used in MF-BPR).
        - MLP_model (nn.Module, optional): Pre-trained MLP model (not used in MF-BPR).
        """
        super(MF_BPR, self).__init__()
        self.dropout = dropout
        self.model = model
        self.MLP_model = MLP_model
        
        # Embedding layers for MF-BPR
        self.embed_user_MLP = nn.Embedding(user_num, factor_num)
        self.embed_item_MLP = nn.Embedding(item_num, factor_num)

        # Initialize weights
        self._init_weight_()

    def _init_weight_(self):
        """
        Initialize weights for user and item embeddings.
        """
        nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
        nn.init.normal_(self.embed_item_MLP.weight, std=0.01)

    def forward_one_item(self, user, item):
        """
        Forward pass for predicting score for a single user-item pair.

        Args:
        - user (Tensor): Tensor of user indices.
        - item (Tensor): Tensor of item indices.

        Returns:
        - pred (Tensor): Predicted score for the user-item pair.
        """
        embed_user_MLP = self.embed_user_MLP(user)
        embed_item_MLP = self.embed_item_MLP(item)

        # Element-wise multiplication
        pred = torch.mul(embed_user_MLP, embed_item_MLP)
        # Sum along the factor_num dimension
        pred = torch.sum(pred, 1)

        return pred.view(-1)

    def forward(self, user, item_i, item_j):
        """
        Forward pass for a triplet of user and two items (positive and negative).

        Args:
        - user (Tensor): Tensor of user indices.
        - item_i (Tensor): Tensor of positive item indices.
        - item_j (Tensor): Tensor of negative item indices.

        Returns:
        - prediction_i (Tensor): Predicted score for the user-positive item pair.
        - prediction_j (Tensor): Predicted score for the user-negative item pair.
        """
        prediction_i = self.forward_one_item(user, item_i)
        prediction_j = self.forward_one_item(user, item_j)

        return prediction_i, prediction_j
		
class MF_MACR(nn.Module):
    def __init__(self, user_num, item_num, factor_num, num_layers,
                 dropout, model, GMF_model=None, MLP_model=None):
        """
        Initialize Matrix Factorization with MACR (Macro-Coef for user and item) model.

        Args:
        - user_num (int): Number of users in the dataset.
        - item_num (int): Number of items in the dataset.
        - factor_num (int): Number of latent factors to use for user and item embeddings.
        - num_layers (int): Number of layers in the MLP part of the model (not used in MF-MACR).
        - dropout (float): Dropout probability (not used in MF-MACR).
        - model (str): Type of model to use ('MLP', 'GMF', or others, but MF-MACR only supports 'MLP').
        - GMF_model (nn.Module, optional): Pre-trained GMF model (not used in MF-MACR).
        - MLP_model (nn.Module, optional): Pre-trained MLP model (not used in MF-MACR).
        """
        super(MF_MACR, self).__init__()
        self.dropout = dropout
        self.model = model
        self.MLP_model = MLP_model
        
        # Embedding layers for user and item in MF-MACR
        self.embed_user_MLP = nn.Embedding(user_num, factor_num)
        self.embed_item_MLP = nn.Embedding(item_num, factor_num)
        
        # Macro-coefficient embeddings for users and items
        self.macr_user = nn.Embedding(user_num, 1)
        self.macr_item = nn.Embedding(item_num, 1)

        # Initialize weights
        self._init_weight_()

    def _init_weight_(self):
        """
        Initialize weights for embeddings and macro-coefficients.
        """
        nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
        nn.init.normal_(self.embed_item_MLP.weight, std=0.01)
        nn.init.normal_(self.macr_user.weight, std=0.01)
        nn.init.normal_(self.macr_item.weight, std=0.01)
        
    def forward_one_item(self, user, item):
        """
        Forward pass for predicting score for a single user-item pair.

        Args:
        - user (Tensor): Tensor of user indices.
        - item (Tensor): Tensor of item indices.

        Returns:
        - pred (Tensor): Predicted score for the user-item pair.
        """
        embed_user_MLP = self.embed_user_MLP(user)
        embed_item_MLP = self.embed_item_MLP(item)

        # Element-wise multiplication
        pred = torch.mul(embed_user_MLP, embed_item_MLP)
        # Sum along the factor_num dimension
        pred = torch.sum(pred, 1)

        return pred.view(-1)

    def forward(self, user, item_i, item_j):
        """
        Forward pass for a triplet of user and two items (positive and negative).

        Args:
        - user (Tensor): Tensor of user indices.
        - item_i (Tensor): Tensor of positive item indices.
        - item_j (Tensor): Tensor of negative item indices.

        Returns:
        - prediction_i (Tensor): Predicted score for the user-positive item pair.
        - prediction_j (Tensor): Predicted score for the user-negative item pair.
        """
        prediction_i = self.forward_one_item(user, item_i)
        prediction_j = self.forward_one_item(user, item_j)

        return prediction_i, prediction_j