import numpy as np
import logging
import copy
import numpy as np
import random
import sys
sys.path.append('./')
sys.path.append('..')
sys.path.append('.')
from utils.space_utils import *
import itertools

class Categorical(object):


    def __init__(self, categories):
        self.d = len(categories)
        self.C = categories
        self.Cmax = np.max(categories)
        self.theta = np.zeros((self.d, self.Cmax))

        for i in range(self.d):
            self.theta[i, :self.C[i]] = 1./self.C[i]

        for i in range(self.d):
            self.theta[i, self.C[i]:] = 0.


    def sampling_lam(self, lam):

        rand = np.random.rand(lam, self.d, 1)    
        cum_theta = self.theta.cumsum(axis=1)    
        X = (cum_theta - self.theta <= rand) & (rand < cum_theta)
        return X

    def sampling(self):

        rand = np.random.rand(self.d, 1)   
        cum_theta = self.theta.cumsum(axis=1)    

    
        x = (cum_theta - self.theta <= rand) & (rand < cum_theta)
        return x

    def sampling_index(self):

        index_list = []
        for prob in self.theta:
            index_list.append(np.random.choice(a=list(range(prob.shape[0])), p=prob))
        return np.array(index_list)

    def mle(self):

        m = self.theta.argmax(axis=1)
        x = np.zeros((self.d, self.Cmax))
        for i, c in enumerate(m):
            x[i, c] = 1
        return x

    def loglikelihood(self, X):

        return (X * np.log(self.theta)).sum(axis=2).sum(axis=1)

    def log_header(self):
        header_list = []
        for i in range(self.d):
            header_list += ['theta%d_%d' % (i, j) for j in range(self.C[i])]
        return header_list

    def log(self):
        theta_list = []
        for i in range(self.d):
            theta_list += ['%f' % self.theta[i, j] for j in range(self.C[i])]
        return theta_list

    def load_theta_from_log(self, theta):
        self.theta = np.zeros((self.d, self.Cmax))
        k = 0
        for i in range(self.d):
            for j in range(self.C[i]):
                self.theta[i, j] = theta[k]
                k += 1

    def print_theta(self, logger):

        org_formatters = []
        for handler in logger.handlers:
            org_formatters.append(handler.formatter)
            handler.setFormatter(logging.Formatter("%(message)s"))

        logger.info("####### theta #######")
        logger.info("# Theta value")
        for alpha in self.theta:
            logger.info(alpha)
        logger.info("#####################")

     
        for handler, formatter in zip(logger.handlers, org_formatters):
            handler.setFormatter(formatter)
        
    def get_infoentropy(self):

        info = np.zeros((2, self.Cmax))
        info[0] = self.theta[0]
        info[1] = np.sum(self.theta[1:], axis = 0)/ (self.d - 1)

        result = 0
        for i,j in itertools.product(range(4),range(info[1].size)):

            result = result - info[0][i]*info[1][j]*np.log(info[0][i]*info[1][j])
        return result

class SNG:

    def __init__(self, categories, delta_init=1., lam=2, init_theta=None, max_mize=True):


        self.p_model = Categorical(categories)

        self.p_model.C = np.array(self.p_model.C)
        self.valid_d = len(self.p_model.C[self.p_model.C > 1])

        if init_theta is not None:
            self.p_model.theta = init_theta


        self.delta = delta_init
        self.lam = lam  
        self.eps = self.delta
        self.sample = []
        self.objective = []
        self.maxmize = -1 if max_mize else 1

    def get_lam(self):
        return self.lam

    def get_delta(self):
        return self.delta

    def record_information(self, sample, objective):
        self.sample.append(sample)
        self.objective.append(objective*self.maxmize)

    def sampling(self):
        rand = np.random.rand(self.p_model.d, 1)  
        cum_theta = self.p_model.theta.cumsum(axis=1) 


        c = (cum_theta - self.p_model.theta <= rand) & (rand < cum_theta)
        return c

    def sampling_index(self):
        return one_hot_to_index(np.array(self.sampling()))
    
    def sampling_subspace(self, search_space):

        layer_num_seq = []
        for layer_num in search_space["bert_layer_num_choice"]:
            layer_num_seq.append(layernum2number(layer_num))
        layer_num_seq.sort()
        encoder_seq = []
        for head_num, ffn_size in itertools.product(search_space["bert_head_choice"], search_space["bert_intermediate_choice"]):
            encoder_seq.append(layer_arc2number(head_num, ffn_size))
        encoder_seq.sort()


        theta_layer_num = np.zeros(len(layer_num_seq))
        sum_layer_pro = 0
        for i in layer_num_seq:
            sum_layer_pro += self.p_model.theta[0,i]
        for k,v in enumerate(layer_num_seq):
            theta_layer_num[k] = self.p_model.theta[0,v]/sum_layer_pro

    
        theta_encoder_arc = self.p_model.theta[1:,encoder_seq]
        sum_theta_encoder_arc = np.sum(theta_encoder_arc, axis = 1)
        sum_theta_encoder_arc = np.expand_dims(sum_theta_encoder_arc, axis=1).repeat(len(encoder_seq), axis=1)
        theta_encoder_arc = np.divide(theta_encoder_arc, sum_theta_encoder_arc)

 
        result_index=[]

  
        layer_index = np.random.choice(a=list(range(theta_layer_num.shape[0])), p=theta_layer_num)
        layer_num = layer_num_seq[layer_index]
        result_index.append(layer_num)
        
    
        encoder_arc_index = []
        for prob in theta_encoder_arc:
            encoder_arc_index.append(np.random.choice(a=list(range(prob.shape[0])), p=prob))
        encoder_arc_num = [encoder_seq[i] for i in encoder_arc_index]
        result_index += encoder_arc_num

  
        for fresh_layer in range(number2layernum(layer_num)+1,13):
            result_index[fresh_layer] = random.choice(list(range(12)))
        
        result = index_to_one_hot(np.array(result_index), 12)
        return result
            

    def mle(self):

        m = self.p_model.theta.argmax(axis=1)
        x = np.zeros((self.p_model.d, self.p_model.Cmax))
        for i, c in enumerate(m):
            x[i, c] = 1
        return x

    def update(self):
        if len(self.sample) == self.lam:
            objective = np.array(self.objective)
            sample_array = np.array(self.sample)
            self.update_function(sample_array, objective)
            self.sample = []
            self.objective = []

    def update_function(self, c_one, fxc, range_restriction=True):

        aru, idx = self.utility(fxc)
        if np.all(aru == 0):

            return

        ng = np.mean(aru[:, np.newaxis, np.newaxis] * (c_one[idx] - self.p_model.theta), axis=0)

        sl = []
        for i, K in enumerate(self.p_model.C):
            theta_i = self.p_model.theta[i, :K - 1]
            theta_K = self.p_model.theta[i, K - 1]
            s_i = 1. / np.sqrt(theta_i) * ng[i, :K - 1]
            s_i += np.sqrt(theta_i) * ng[i, :K - 1].sum() / (theta_K + np.sqrt(theta_K))
            sl += list(s_i)
        sl = np.array(sl)

        pnorm = np.sqrt(np.dot(sl, sl)) + 1e-8
        self.eps = self.delta / pnorm
        self.p_model.theta += self.eps * ng

        for i in range(self.p_model.d):
            ci = self.p_model.C[i]

            theta_min = 1. / (self.valid_d * (ci - 1)) if range_restriction and ci > 1 else 0.
            self.p_model.theta[i, :ci] = np.maximum(self.p_model.theta[i, :ci], theta_min)
            theta_sum = self.p_model.theta[i, :ci].sum()
            tmp = theta_sum - theta_min * ci
            self.p_model.theta[i, :ci] -= (theta_sum - 1.) * (self.p_model.theta[i, :ci] - theta_min) / tmp

            self.p_model.theta[i, :ci] /= self.p_model.theta[i, :ci].sum()

    @staticmethod
    def utility(f, rho=0.25, negative=True):

        eps = 1e-14
        idx = np.argsort(f)
        lam = len(f)
        mu = int(np.ceil(lam * rho))
        _w = np.zeros(lam)
        _w[:mu] = 1 / mu
        _w[lam - mu:] = -1 / mu if negative else 0
        w = np.zeros(lam)
        istart = 0
        for i in range(f.shape[0] - 1):
            if f[idx[i + 1]] - f[idx[i]] < eps * f[idx[i]]:
                pass
            elif istart < i:
                w[istart:i + 1] = np.mean(_w[istart:i + 1])
                istart = i + 1
            else:
                w[i] = _w[i]
                istart = i + 1
        w[istart:] = np.mean(_w[istart:])
        return w, idx

    def log_header(self, theta_log=False):
        header_list = ['delta', 'eps', 'theta_converge']
        if theta_log:
            for i in range(self.p_model.d):
                header_list += ['theta%d_%d' % (i, j) for j in range(self.C[i])]
        return header_list

    def log(self, theta_log=False):
        log_list = [self.delta, self.eps, self.p_model.theta.max(axis=1).mean()]

        if theta_log:
            for i in range(self.p_model.d):
                log_list += ['%f' % self.p_model.theta[i, j] for j in range(self.C[i])]
        return log_list

    def load_theta_from_log(self, theta_log):
        self.p_model.theta = np.zeros((self.p_model.d, self.p_model.Cmax))
        k = 0
        for i in range(self.p_model.d):
            for j in range(self.p_model.C[i]):
                self.p_model.theta[i, j] = theta_log[k]
                k += 1
