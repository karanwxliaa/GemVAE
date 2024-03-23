import tensorflow.compat.v1 as tf
import tensorflow as tf
import tensorflow.compat.v1 as v1

class LinBnDrop(tf.keras.Sequential):
    def __init__(self, n_in, n_out, bn=True, p=0., act=None, lin_first=True):
        layers = []
        if bn:
            layers.append(tf.keras.layers.BatchNormalization())
        if p != 0:
            layers.append(tf.keras.layers.Dropout(p))
        lin = [tf.keras.layers.Dense(n_out, use_bias=not bn)]
        if act is not None:
            layers.append(act)
        layers = lin + layers if lin_first else layers + lin
        super(LinBnDrop, self).__init__(layers)


class GATE():
    def __init__(self, hidden_dims1, hidden_dims2,z_dim=30,alpha=0.3, nonlinear=True, weight_decay=0.0001, num_hidden=256, num_proj_hidden=256, tau=0.5,kl_loss = 0,contrastive_loss = 10,recon_loss = 1,weight_decay_loss = 1,recon_loss_type = "MSE"):
        self.n_layers1 = len(hidden_dims1) - 1
        self.n_layers2 = len(hidden_dims2) - 1
        self.alpha = alpha
        self.W1, self.v1, self.prune_v1 = self.define_weights1(hidden_dims1, self.n_layers1)
        self.W2, self.v2, self.prune_v2 = self.define_weights2(hidden_dims2, self.n_layers2)
        self.C1 = {}
        self.C2 = {}
        self.prune_C1 = {}
        self.prune_C2 = {}
        self.nonlinear = nonlinear
        self.weight_decay = weight_decay
        self.z_dim = z_dim
        self.fc_mu = LinBnDrop(z_dim,z_dim, p=0.2)
        self.fc_var = LinBnDrop(z_dim,z_dim, p=0.2)

        self.tau = tau
        self.fc1 = tf.keras.layers.Dense(num_proj_hidden, activation='elu')
        self.fc2 = tf.keras.layers.Dense(num_hidden)
        self.dropout_rate = 0.1

        self.kl_loss = kl_loss
        self.contrastive_loss = contrastive_loss
        self.recon_loss = recon_loss
        self.weight_decay_loss = weight_decay_loss
        self.recon_loss_type = recon_loss_type

        # Decoder 1
        self.W_dec1 = {}
        for layer in range(self.n_layers1 - 1, -1, -1):
            self.W_dec1[layer] = tf.Variable(tf.random.normal([hidden_dims1[layer+1], hidden_dims1[layer]]))

        # Decoder 2
        self.W_dec2 = {}
        for layer in range(self.n_layers2 - 1, -1, -1):
            self.W_dec2[layer] = tf.Variable(tf.random.normal([hidden_dims2[layer+1], hidden_dims2[layer]]))

    def __call__(self, A1,A2 ,prune_A1,prune_A2, X1,X2):
        # Encoder 1
        H1 = X1
        for layer in range(self.n_layers1):
            H1 = self.__encoder1(A1, prune_A1, H1, layer)
            if self.nonlinear:
                if layer != self.n_layers1 - 1:
                    H1 = tf.nn.elu(H1)

        # Encoder 2
        H2 = X2
        for layer in range(self.n_layers2):
            H2 = self.__encoder2(A2, prune_A2, H2, layer)
            if self.nonlinear:
                if layer != self.n_layers2 - 1:
                    H2 = tf.nn.elu(H2)

        con_loss = self.con_loss(H1, H2, batch_size=0)

        self.c_loss = con_loss

        # Concatenate encoder outputs
        H = tf.concat([H1, H2], axis=1)

        # Call the third encoder
        global latent_rep 
        H = self.__encoder3(H)

        #Latent space using a varational auto encoder
        mu = self.fc_mu(H)
        var = self.fc_var(H)
        H = self.reparameterize(mu, var)
        
        latent_rep = H

        # KL Divergence Loss
        kl_divergance_loss = -0.5 * tf.reduce_sum(1 + var - tf.square(mu) - tf.exp(var), axis=1)
        kl_divergance_loss = tf.reduce_mean(kl_divergance_loss)


        temp=H
        H1=temp
        # Decoder 1
        for layer in range(self.n_layers1 - 1, -1, -1):
            H1 = self.__decoder1(H1, layer)
            if self.nonlinear:
                if layer != 0:
                    H1 = tf.nn.elu(H1)
        X1_ = H1

        H2=temp
        # Decoder 2
        for layer1 in range(self.n_layers2 - 1, -1, -1):
            H2 = self.__decoder2(H2, layer1)
            if self.nonlinear:
                if layer1 != 0:
                    H2 = tf.nn.elu(H2)
        X2_ = H2

        # Loss calculation
        # Calculating inputs for the ZINB loss
        # Data normalization (optional)
        X1_, X2_ = v1.nn.softmax(X1_), v1.nn.softmax(X2_)

        if self.recon_loss_type == 'ZINB':
            #USING ZINB FOR LOSS CALC
            # Estimate library size as in reference code
            log_library_size1 = v1.math.log(v1.reduce_sum(X1_, axis=-1) + 1)
            #log_library_size2 = v1.math.log(v1.reduce_sum(X2_, axis=-1) + 1)

            library_size_mean1 = v1.reduce_mean(log_library_size1)
            #library_size_variance1 = v1.math.reduce_variance(log_library_size1)

            #library_size_mean2 = v1.reduce_mean(log_library_size2)
            #library_size_variance2 = v1.math.reduce_variance(log_library_size2)

            self.x_post_r1 = v1.random.normal(shape=[X1_.shape[-1]], dtype=v1.float32)
            #self.x_post_r2 = v1.random.normal(shape=[X2_.shape[-1]], dtype=v1.float32)

            # They used an additional layer between decoder and zinb loss
            # You can consider adding it if the performance is not satisfactory 

            x_post_scale1 = v1.exp(library_size_mean1) * X1_
            #x_post_scale2 = v1.exp(library_size_mean2) * X2_

            local_dispersion1 = v1.exp(self.x_post_r1)
            #local_dispersion2 = v1.exp(self.x_post_r2)

            x_post_dropout1 = v1.nn.dropout(X1_, self.dropout_rate)
            #x_post_dropout2 = v1.nn.dropout(X2_, self.dropout_rate)

            # ZINB Loss calculation
            zinb_loss1 = self.zinb_model(X1, x_post_scale1, local_dispersion1, x_post_dropout1)
            #zinb_loss2 = self.zinb_model(X2, x_post_scale2, local_dispersion2, x_post_dropout2)
            
            # Calculate the mean of zinb_loss1 and reconstruction_loss

            
            rloss = tf.reduce_mean(zinb_loss1) 
            #rloss += tf.reduce_mean(zinb_loss2)
            rloss*=-0.5

            
        else:
            #using MSE
            print("Using MSE for gene")
            rloss = tf.sqrt(tf.reduce_sum(tf.pow(X1 - X1_, 2)))
            
        #MSE always for protien 
        rloss += tf.sqrt(tf.reduce_sum(tf.pow(X2 - X2_, 2)))
  



        weight_decay_loss = 0
        for layer in range(self.n_layers1):
            weight_decay_loss += tf.multiply(tf.nn.l2_loss(self.W1[layer]), self.weight_decay, name='weight_loss')
        for layer in range(self.n_layers2):
            weight_decay_loss += tf.multiply(tf.nn.l2_loss(self.W2[layer]), self.weight_decay, name='weight_loss')
        for layer in range(self.n_layers1):
            weight_decay_loss += tf.multiply(tf.nn.l2_loss(self.W_dec1[layer]), self.weight_decay, name='weight_loss')
        for layer in range(self.n_layers2):
            weight_decay_loss += tf.multiply(tf.nn.l2_loss(self.W_dec2[layer]), self.weight_decay, name='weight_loss')

        # Total loss
        print("Loss weights are = ",self.contrastive_loss,self.recon_loss,self.weight_decay_loss,self.kl_loss)
        self.loss = (self.contrastive_loss*con_loss) + (self.recon_loss*rloss) + (self.weight_decay_loss*weight_decay_loss) + (self.kl_loss*kl_divergance_loss)

        if self.alpha == 0:
            print("\n\nAlpha = 0")
            self.Att_l = {'C1': self.C1, 'C2': self.C2}
        else:
            self.Att_l = {'C1': self.C1, 'C2': self.C2, 'prune_C1': self.prune_C1, 'prune_C2': self.prune_C2}
            

        return self.c_loss, self.loss, latent_rep, self.Att_l, X1_, X2_

    
    # Define the zinb_model loss function
    def zinb_model(self, x, mean, inverse_dispersion, logit, eps=1e-4):
        expr_non_zero = - v1.nn.softplus(- logit) \
                        + v1.log(inverse_dispersion + eps) * inverse_dispersion \
                        - v1.log(inverse_dispersion + mean + eps) * inverse_dispersion \
                        - x * v1.log(inverse_dispersion + mean + eps) \
                        + x * v1.log(mean + eps) \
                        - v1.lgamma(x + 1) \
                        + v1.lgamma(x + inverse_dispersion) \
                        - v1.lgamma(inverse_dispersion) \
                        - logit 
        
        expr_zero = - v1.nn.softplus( - logit) \
                    + v1.nn.softplus(- logit + v1.log(inverse_dispersion + eps) * inverse_dispersion \
                                    - v1.log(inverse_dispersion + mean + eps) * inverse_dispersion) 
        
        template = v1.cast(v1.less(x, eps), v1.float32)
        expr =  v1.multiply(template, expr_zero) + v1.multiply(1 - template, expr_non_zero)
        
        return v1.reduce_sum(expr, axis=-1)
    
    def projection(self, z):
        z = self.fc1(z)
        return self.fc2(z)

    def sim(self, z1, z2):
        z1 = tf.nn.l2_normalize(z1, axis=1)
        z2 = tf.nn.l2_normalize(z2, axis=1)
        return tf.matmul(z1, z2, transpose_b=True)

    def semi_loss(self, z1, z2):
        f = lambda x: tf.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))
        diag_ref_sim = tf.linalg.diag_part(refl_sim)

        return -tf.math.log(
            tf.linalg.diag_part(between_sim)
            / (tf.reduce_sum(refl_sim, axis=1) + tf.reduce_sum(between_sim, axis=1) - diag_ref_sim))

    def batched_semi_loss(self, z1, z2, batch_size):
        num_nodes = tf.shape(z1)[0]
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: tf.exp(x / self.tau)
        indices = tf.range(0, num_nodes)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(tf.gather(z1, mask), z1))  # [B, N]
            between_sim = f(self.sim(tf.gather(z1, mask), z2))  # [B, N]

            losses.append(-tf.math.log(
                tf.linalg.diag_part(tf.gather(between_sim, mask, batch_dims=1))
                / (tf.reduce_sum(refl_sim, axis=1) + tf.reduce_sum(between_sim, axis=1)
                   - tf.linalg.diag_part(tf.gather(refl_sim, mask, batch_dims=1)))))

        return tf.concat(losses, axis=0)

    def con_loss(self, z1, z2, mean=True, batch_size=0):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size == 0:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = tf.reduce_mean(ret) if mean else tf.reduce_sum(ret)

        return ret
    

    def reparameterize(self, mu, logvar):
        std = tf.exp(0.5 * logvar)
        eps = tf.random.normal(shape=tf.shape(std))
        return eps * std + mu

    def __encoder1(self, A, prune_A1, H, layer):
        ##print('enc1 = ',H)
        H = tf.matmul(H, self.W1[layer])
        if layer == self.n_layers1 - 1:
            return H
        self.C1[layer] = self.graph_attention_layer(A, H, self.v1[layer], layer)
        if self.alpha == 0:
            return tf.sparse.sparse_dense_matmul(self.C1[layer], H)
        else:
            self.prune_C1[layer] = self.graph_attention_layer(prune_A1, H, self.prune_v1[layer], layer)
            return (1 - self.alpha) * tf.sparse.sparse_dense_matmul(self.C1[layer], H) + self.alpha * tf.sparse.sparse_dense_matmul(
                self.prune_C1[layer], H)
        
        

    def __encoder2(self, A, prune_A2, H, layer):
        #print('enc2 = ',H)
        H = tf.matmul(H, self.W2[layer])
        if layer == self.n_layers2 - 1:
            return H
        self.C2[layer] = self.graph_attention_layer(A, H, self.v2[layer], layer)
        if self.alpha == 0:
            return tf.sparse.sparse_dense_matmul(self.C2[layer], H)
        else:
            self.prune_C2[layer] = self.graph_attention_layer(prune_A2, H, self.prune_v2[layer], layer)
            return (1 - self.alpha) * tf.sparse.sparse_dense_matmul(self.C2[layer], H) + self.alpha * tf.sparse.sparse_dense_matmul(
                self.prune_C2[layer], H)
    
    def __decoder1(self, H, layer):
        #print('dec1 = ',H)
        H = tf.matmul(H, self.W1[layer], transpose_b=True)
        if layer == 0:

            return H
        if self.alpha == 0:

            return tf.sparse.sparse_dense_matmul(self.C1[layer-1], H)
        else:

            return (1 - self.alpha) * tf.sparse.sparse_dense_matmul(self.C1[layer-1], H) + self.alpha * tf.sparse.sparse_dense_matmul(
                self.prune_C1[layer-1], H)
        
    def __decoder2(self, H, layer):
        #print('dec2 = ',H)
        H = tf.matmul(H, self.W2[layer], transpose_b=True)
        if layer == 0:

            return H
        if self.alpha == 0:
            return tf.sparse.sparse_dense_matmul(self.C2[layer-1], H)
        
        else:

            return (1 - self.alpha) * tf.sparse.sparse_dense_matmul(self.C2[layer-1], H) + self.alpha * tf.sparse.sparse_dense_matmul(
                self.prune_C2[layer-1], H)
        

    def __encoder3(self, H):
        #print('enc3 = ',H)
        H = tf.keras.layers.Dense(self.z_dim)(H)
        #print('LATENT = ',H)
        return H


    

    def define_weights1(self,hidden_dims,n_layers):
        W = {}
        ##print('TOTAL LEYRS = ',n_layers)
        #n_layers=len(n_layers)-1
        #print('n_layers gene = ',n_layers)
        #print('Hidden dim gene = ',hidden_dims)

        for i in range(n_layers):
            W[i] = v1.get_variable("W%s" % i, shape=(hidden_dims[i], hidden_dims[i+1]))

        Ws_att = {}
        for i in range(n_layers-1):
            V= {}
            V[0] = v1.get_variable("V%s_0" % i, shape=(hidden_dims[i+1], 1))
            V[1] = v1.get_variable("V%s_1" % i, shape=(hidden_dims[i+1], 1))

            Ws_att[i] = V
        if self.alpha == 0:
            return W, Ws_att, None
        prune_Ws_att = {}
        for i in range(n_layers-1):
            prune_V = {}
            prune_V[0] = v1.get_variable("prune_V%s_0" % i, shape=(hidden_dims[i+1], 1))
            prune_V[1] = v1.get_variable("prune_V%s_1" % i, shape=(hidden_dims[i+1], 1))

            prune_Ws_att[i] = prune_V

        return W, Ws_att, prune_Ws_att
    
    def define_weights2(self,hidden_dims,n_layers):
        w = {}
        ##print('TOTAL LEYRS = ',n_layers)
        #n_layers=len(n_layers)-1
        #print('n_layers protein = ',n_layers)
        #print('Hidden dim protein = ',hidden_dims)

        for i in range(n_layers):
            w[i] = v1.get_variable("w%s" % i, shape=(hidden_dims[i], hidden_dims[i+1]))

        ws_att = {}
        for i in range(n_layers-1):
            v = {}
            v[0] = v1.get_variable("v%s_0" % i, shape=(hidden_dims[i+1], 1))
            v[1] = v1.get_variable("v%s_1" % i, shape=(hidden_dims[i+1], 1))

            ws_att[i] = v
        if self.alpha == 0:
            return w, ws_att, None
        prune_ws_att = {}
        for i in range(n_layers-1):
            prune_v = {}
            prune_v[0] = v1.get_variable("prune_v%s_0" % i, shape=(hidden_dims[i+1], 1))
            prune_v[1] = v1.get_variable("prune_v%s_1" % i, shape=(hidden_dims[i+1], 1))

            prune_ws_att[i] = prune_v

        return w, ws_att, prune_ws_att
    


    def graph_attention_layer(self, A, M, v, layer):

        with v1.variable_scope("layer_%s" % layer):
            f1 = tf.matmul(M, v[0])
            f1 = A * f1
            f2 = tf.matmul(M, v[1])
            f2 = A * tf.transpose(f2, [1, 0])
            logits = v1.sparse_add(f1, f2)

            unnormalized_attentions = tf.SparseTensor(indices=logits.indices,
                                         values=tf.nn.sigmoid(logits.values),
                                         dense_shape=logits.dense_shape)
            attentions = v1.sparse_softmax(unnormalized_attentions)

            attentions = tf.SparseTensor(indices=attentions.indices,
                                         values=attentions.values,
                                         dense_shape=attentions.dense_shape)

            return attentions