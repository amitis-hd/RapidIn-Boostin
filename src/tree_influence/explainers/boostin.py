
import time
import numpy as np
from .base import Explainer
from .parsers import util
import torch.nn.functional as F
from torch.utils.data import default_collate
import torch

from sklearn.preprocessing import normalize
import math, random

##Added a normalization step

class BoostIn(Explainer):
    """
    Explainer that adapts the TracIn method to tree ensembles.

    Local-Influence Semantics
        - Inf.(x_i, x_t) = sum grad(x_t) * leaf_der w.r.t. x_i * learning_rate over all boosts.
        - Pos. value means a decrease in test loss (a.k.a. proponent, helpful).
        - Neg. value means an increase in test loss (a.k.a. opponent, harmful).

    Reference
        - https://github.com/frederick0329/TracIn

    Paper
        - https://arxiv.org/abs/2002.08484

    Note
        - Only support GBDTs.
    """
    def __init__(self, logger=None):
        """
        Input
            logger: object, If not None, output to logger.
        """
        self.logger = logger

    def fit(self, model, X, y , oporp_eng , gpu):
        """
        - Convert model to internal standardized tree structure.
        - Precompute gradients and leaf indices for each x in X.

        Input
            model: tree ensemble.
            X: 2d array of train examples.
            y: 1d array of train targets.
        """
        X, y = util.convert_to_np(X, y)
        super().fit(model, X, y)
        X, y = util.check_data(X, y, objective=self.model_.objective)

        assert self.model_.tree_type != 'rf', 'RF not supported for BoostIn'

        self.n_train_ = X.shape[0]
        self.loss_fn_ = util.get_loss_fn(self.model_.objective, self.model_.n_class_, self.model_.factor)

        self.train_leaf_dvs_ = self._compute_leaf_derivatives(X, y , oporp_eng , gpu )  # (X.shape[0], n_boost, n_class)
        self.train_leaf_idxs_ = self.model_.apply(X)  # shape=(X.shape[0], no. boost, no. class)

        return self

    def get_local_influence(self, X, y, oporp_eng , verbose=1 , gpu = 0):
        """
        - Computes effect of each train example on the loss of the test example.

        Input
            X: 2d array of test data.
            y: 2d array of test targets.
            verbose: verbosity.

        Return
            - 2d array of shape=(no. train, X.shape[0]).
                * Array is returned in the same order as the training data.

        Note
            - Attribute train attribution to the test loss ONLY if the train example
                is in the same leaf(s) as the test example.
        """
        start = time.time()

        X, y = util.check_data(X, y, objective=self.model_.objective)

        # result container, shape=(X.shape[0], no. train, no. class)
        influence = np.zeros((self.n_train_, X.shape[0]), dtype=util.dtype_t)

        # get change in leaf derivatives and test prediction derivatives
        train_leaf_dvs = self.train_leaf_dvs_  # (no. train, no. boost, no. class)
        test_gradients = self._compute_gradients(X, y , oporp_eng)  # shape=(X.shape[0], no. boost, no. class)

        # get leaf indices each example arrives in
        train_leaf_idxs = self.train_leaf_idxs_  # shape=(no. train, no. boost, no. class)
        test_leaf_idxs = self.model_.apply(X)  # shape=(X.shape[0], no. boost, no. class)

        # compute attributions for each test example
        for i in range(X.shape[0]):
            mask = np.where(train_leaf_idxs == test_leaf_idxs[i], 1, 0)  # shape=(no. train, no. boost, no. class)
            mask_tensor = torch.tensor(mask , dtype=torch.float32, device = torch.device(f'cuda:{gpu}'))
            #prod = torch.matmul(train_leaf_dvs , test_gradients[i])* mask  # shape=(no. train, no. boost, no. class)
            
            #prod = torch.einsum('ijk,jk->ijk', train_leaf_dvs, test_gradients[i]) * mask  # shape=(no. train, no. boost, no. class)

            
            prod = (train_leaf_dvs * test_gradients[i]) * mask_tensor

            # sum over boosts and classes
            influence[:, i] = prod.sum(dim=(1, 2)).cpu().numpy()  # shape=(no. train,)

            # progress
            if i > 0 and (i + 1) % 100 == 0 and self.logger and verbose:
                self.logger.info(f'[INFO - BoostIn] No. finished: {i+1:>10,} / {X.shape[0]:>10,}, '
                                 f'cum. time: {time.time() - start:.3f}s')

        return influence

    # private
    def _compute_gradients(self, X, y , oporp_eng , gpu = 0):
        """
        - Compute gradients for all train instances across all boosting iterations.

        Input
            X: 2d array of train examples.
            y: 1d array of train targets.

        Return
            - 3d array of shape=(X.shape[0], no. boost, no. class).
        """
        trees = self.model_.trees
        n_boost = self.model_.n_boost_
        n_class = self.model_.n_class_
        bias = self.model_.bias
        

        current_approx = np.tile(bias, (X.shape[0], 1)).astype(util.dtype_t)  # shape=(X.shape[0], no. class)
        gradients = torch.zeros((X.shape[0], n_boost, n_class),  device=torch.device(f'cuda:{gpu}') )  # shape=(X.shape[0], no. boost, no. class)

        oporp_eng_vals = []
        #trying to connect things here


        # compute gradients for each boosting iteration
        for boost_idx in range(n_boost):
 
            gradients[:, boost_idx, :] = torch.tensor(self.loss_fn_.gradient(y, current_approx),  dtype=torch.float32, device=torch.device(f'cuda:{gpu}'))  # shape=(X.shape[0], no. class)

           
            #normalizing
            #gradients[:, boost_idx, :] = torch.tensor(gradients[:, boost_idx, :], dtype=torch.float32)
            gradients[:, boost_idx, :] = self.normalize_tensor(gradients[:, boost_idx, :])

    
            #RapidGrad
            #K = len(gradients[:, boost_idx, :][0])

            #oporp_eng_vals.append(oporp_eng(gradients[:, boost_idx, :], K ))
            #gradients[:, boost_idx, :] = oporp_eng(gradients[:, boost_idx, :], K )

            # update approximation
            for class_idx in range(n_class):
                current_approx[:, class_idx] += trees[boost_idx, class_idx].predict(X)
        #print(f"gradients at 0: {gradients[0]}")
        #print(f"oporp vals grad: {oporp_eng_vals}")
        return gradients 

    def _compute_leaf_derivatives(self, X, y , oporp_eng , gpu = 0):
        """
        - Compute leaf derivatives for all train instances across all boosting iterations.

        Input
            X: 2d array of train examples.
            y: 1d array of train targets.

        Return
            - 3d array of shape=(X.shape[0], no. boost, no. class).

        Note
            - It is assumed that the leaf estimation method is 'Newton'.
        """
        n_train = X.shape[0]

        trees = self.model_.trees
        n_boost = self.model_.n_boost_
        n_class = self.model_.n_class_
        bias = self.model_.bias
        l2_leaf_reg = self.model_.l2_leaf_reg
        lr = self.model_.learning_rate

        # get leaf info
        leaf_counts = self.model_.get_leaf_counts()  # shape=(no. boost, no. class)
        leaf_idxs = self.model_.apply(X)  # shape=(X.shape[0], no. boost, no. class)

        # intermediate container
        current_approx = np.tile(bias, (n_train, 1)).astype(util.dtype_t)  # shape=(X.shape[0], no. class)

        # result container
        leaf_dvs = torch.zeros((n_train, n_boost, n_class), dtype=torch.float32 , device=torch.device(f'cuda:{gpu}'))  # Initialize as torch tensor
        # shape=(X.shape[0], n_boost, n_class)


        #trying to connect things here
        #print(f"Number of GPUs available: {torch.cuda.device_count()}")
        #for i in range(torch.cuda.device_count()):
            #print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
 
        


        # compute gradients for each boosting iteration
        for boost_idx in range(n_boost):
            
            g = self.loss_fn_.gradient(y, current_approx)  # shape=(no. train, no. class)
            h = self.loss_fn_.hessian(y, current_approx)  # shape=(no. train, no. class)

            for class_idx in range(n_class):
                leaf_count = leaf_counts[boost_idx, class_idx]
                leaf_vals = trees[boost_idx, class_idx].get_leaf_values()  # shape=(no. leaves,)

                for leaf_idx in range(leaf_count):
                    leaf_docs = torch.tensor(np.where(leaf_idx == leaf_idxs[:, boost_idx, class_idx])[0], dtype=torch.long, device=torch.device(f'cuda:{gpu}'))
                    leaf_docs_cpu = leaf_docs.cpu()
            

                    # compute leaf derivative w.r.t. each train example in `leaf_docs`
                    numerator = g[leaf_docs_cpu, class_idx] + leaf_vals[leaf_idx] * h[leaf_docs_cpu, class_idx]  # (no. docs,)
                    denominator = np.sum(h[leaf_docs_cpu, class_idx]) + l2_leaf_reg
           
                    leaf_dvs[leaf_docs, boost_idx, class_idx] = self.normalize_tensor(torch.tensor(numerator / denominator * lr , dtype=torch.float32 ,device=torch.device(f'cuda:{gpu}')))  # (no. docs,)
         


                    #normalizing
                    #leaf_dvs[leaf_docs, boost_idx, class_idx] = torch.tensor(leaf_dvs[leaf_docs, boost_idx, class_idx], dtype=torch.float32)
                    #leaf_dvs[leaf_docs, boost_idx, class_idx] = self.normalize(leaf_dvs[leaf_docs, boost_idx, class_idx])
                    #RapidGrad
                    #leaf_dvs[leaf_docs, boost_idx, class_idx] =leaf_dvs[leaf_docs, boost_idx, class_idx].cuda(0)
                    #K = len(leaf_dvs[leaf_docs, boost_idx, class_idx])
                     # RapidGrad
                    #oporp_eng_value = oporp_eng(leaf_dvs[leaf_docs, boost_idx, class_idx], K)
                    #leaf_dvs[leaf_docs, boost_idx, class_idx] = oporp_eng(leaf_dvs[leaf_docs, boost_idx, class_idx], K) 
                    #oporp_eng_values.append(oporp_eng_value)


                # update approximation
                current_approx[:, class_idx] += trees[boost_idx, class_idx].predict(X)

        #print(f"leaf dvs at 0 : {leaf_dvs[0]}")
        #print(f"leaf oporp vals at 0: {oporp_eng_values[0]}")
        return leaf_dvs 

    
    def normalize_tensor(self, tensor, dim=0):
        """
        Normalizes a PyTorch tensor along a specified dimension.

        Input
            tensor: PyTorch tensor to be normalized.
            dim: Dimension along which to normalize.

        Return
            - Normalized tensor.
        """
        return torch.nn.functional.normalize(tensor , p=2 , dim = 0)
    

    
    
    
    

