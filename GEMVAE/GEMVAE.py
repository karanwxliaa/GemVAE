import tensorflow as tf
import scipy.sparse as sp
import numpy as np
from .model import GATE
import tensorflow.compat.v1 as v1
from tqdm import tqdm


class GEMVAE():

    def __init__(self, hidden_dims1, hidden_dims2,z_dim=30, alpha=0, n_epochs=500, lr=0.0001, 
                 gradient_clipping=5, nonlinear=True, weight_decay=0.0001, 
                 verbose=True, random_seed=2020,
                 kl_loss = 0,contrastive_loss = 10,recon_loss = 1,weight_decay_loss = 1,recon_loss_type = "MSE",
                 ):
        
        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)
        self.loss_list = []
        self.lr = lr
        self.n_epochs = n_epochs
        self.gradient_clipping = gradient_clipping
        self.build_placeholders()
        self.verbose = verbose
        self.alpha = alpha

        self.kl_loss = kl_loss
        self.contrastive_loss = contrastive_loss
        self.recon_loss = recon_loss
        self.weight_decay_loss = weight_decay_loss
        self.recon_loss_type = recon_loss_type


        global C
           

        self.gate = GATE(hidden_dims1,hidden_dims2,z_dim,alpha, nonlinear, weight_decay,
                kl_loss = kl_loss,contrastive_loss = contrastive_loss,recon_loss = recon_loss,weight_decay_loss = weight_decay_loss,recon_loss_type = recon_loss_type )
        
        
        self.c_loss, self.loss, self.H, self.C, self.ReX1, self.ReX2 = self.gate(self.A1,self.A2, self.prune_A1,self.prune_A2, self.X1,self.X2)

        self.optimize(self.loss)
        self.build_session()

    def build_placeholders(self):
        self.A1 = v1.sparse_placeholder(dtype=tf.float32)
        self.A2 = v1.sparse_placeholder(dtype=tf.float32)
        self.prune_A1 = v1.sparse_placeholder(dtype=tf.float32)
        self.prune_A2 = v1.sparse_placeholder(dtype=tf.float32)
        self.X1 = v1.placeholder(dtype=tf.float32)
        self.X2 = v1.placeholder(dtype=tf.float32)


    def build_session(self, gpu= True):
        config = v1.ConfigProto()
        config.gpu_options.allow_growth = True
        if gpu == False:
            config.intra_op_parallelism_threads = 0
            config.inter_op_parallelism_threads = 0
        self.session = v1.Session(config=config)
        self.session.run([v1.global_variables_initializer(), v1.local_variables_initializer()])

    def optimize(self, loss):
        optimizer = v1.train.AdamOptimizer(learning_rate=self.lr)
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients, self.gradient_clipping)
        self.train_op = optimizer.apply_gradients(zip(gradients, variables))

    def __call__(self, A1, A2, prune_A1,prune_A2, X1, X2):
        for epoch in range(self.n_epochs):
            self.run_epoch(epoch, A1, A2, prune_A1,prune_A2, X1, X2)




    def run_epoch(self, epoch, A1,A2 ,prune_A1,prune_A2, X1,X2):

        c_loss, loss, _ = self.session.run([self.c_loss, self.loss, self.train_op],
                           feed_dict={self.A1: A1,
                                      self.A2: A2,
                                      self.prune_A1: prune_A1,
                                      self.prune_A2: prune_A2,
                                      self.X1: X1,
                                      self.X2: X2})

        self.loss_list.append(loss)
        if self.verbose:
            print("Epoch: %s, Contrastive Loss: %.4f, Loss: %.4f" % (epoch, c_loss,loss))
            
        return loss

    def infer(self, A1,A2, prune_A1,prune_A2, X1,X2):
        global C
        H, C, ReX1, ReX2 = self.session.run([self.H, self.C, self.ReX1, self.ReX2],
                                    feed_dict={self.A1: A1,
                                               self.A2: A2,
                                               self.prune_A1: prune_A1,
                                               self.prune_A2: prune_A2,
                                               self.X1: X1,
                                               self.X2: X2})


        return H, self.Conbine_Atten_1(C),self.Conbine_Atten_1(C), self.loss_list, ReX1,ReX2

    #Gene
    def Conbine_Atten_1(self, input):
       
        if self.alpha == 0:
            return [sp.coo_matrix((input['C1'][layer][1], (input['C1'][layer][0][:, 0], input['C1'][layer][0][:, 1])), shape=(input['C1'][layer][2][0], input['C1'][layer][2][1])) for layer in input['C1']]
        else:
            
            Att_C = [sp.coo_matrix((input['C1'][layer][1], (input['C1'][layer][0][:, 0], input['C1'][layer][0][:, 1])), shape=(input['C1'][layer][2][0], input['C1'][layer][2][1])) for layer in input['C1']]
            Att_pruneC = [sp.coo_matrix((input['prune_C1'][layer][1], (input['prune_C1'][layer][0][:, 0], input['prune_C1'][layer][0][:, 1])), shape=(input['prune_C1'][layer][2][0], input['prune_C1'][layer][2][1])) for layer in input['prune_C1']]
            return [self.alpha*Att_pruneC[layer] + (1-self.alpha)*Att_C[layer] for layer in input['C1']]

    #Protein
    def Conbine_Atten_2(self, input):
       
        if self.alpha == 0:
            return [sp.coo_matrix((input['C2'][layer][1], (input['C2'][layer][0][:, 0], input['C2'][layer][0][:, 1])), shape=(input['C2'][layer][2][0], input['C2'][layer][2][1])) for layer in input['C2']]
        else:
            
            Att_C = [sp.coo_matrix((input['C2'][layer][1], (input['C2'][layer][0][:, 0], input['C2'][layer][0][:, 1])), shape=(input['C2'][layer][2][0], input['C2'][layer][2][1])) for layer in input['C2']]
            Att_pruneC = [sp.coo_matrix((input['prune_C2'][layer][1], (input['prune_C2'][layer][0][:, 0], input['prune_C2'][layer][0][:, 1])), shape=(input['prune_C2'][layer][2][0], input['prune_C2'][layer][2][1])) for layer in input['prune_C2']]
            return [self.alpha*Att_pruneC[layer] + (1-self.alpha)*Att_C[layer] for layer in input['C2']]