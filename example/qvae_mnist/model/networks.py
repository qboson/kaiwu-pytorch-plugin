# -*- coding: utf-8 -*-
"""
Autoencoders
"""

import torch
import torch.nn as nn

# from .distributions import SpikeAndExponentialSmoother
from copy import copy
import logging
logger = logging.getLogger(__name__)

#Base Class
class Network(nn.Module):
    def __init__(self, node_sequence=None, activation_fct=None, create_module_list=True,**kwargs):
        super(Network, self).__init__(**kwargs)
        self._layers=nn.ModuleList([]) if create_module_list else None
        self._node_sequence=node_sequence
        self._activation_fct=activation_fct

        if self._node_sequence and create_module_list:
            self._create_network()
    
    def encode(self):
        raise NotImplementedError
    
    def decode(self):
        raise NotImplementedError
    
    def _create_network(self):        
        for node in self._node_sequence:
            self._layers.append(
                nn.Linear(node[0],node[1])
            )
        return

    def get_activation_fct(self):        
        return "{0}".format(self._activation_fct).replace("()","")

#Implements encode()
class BasicEncoder(Network):
    def __init__(self, weight_decay=0.0, **kwargs):
        super(BasicEncoder, self).__init__(**kwargs)
        self.weight_decay = weight_decay

    def forward(self, x):
        return self.encode(x)

    def encode(self, x):
        logger.debug("encode")
        for layer in self._layers:
            if self._activation_fct:
                x=self._activation_fct(layer(x))
            else:
                x=layer(x)
        return x

    def get_weight_decay(self) -> torch.Tensor:
        """
        计算当前编码器中所有线性层权重的 L2 正则化损失。
        返回 0.0 如果 weight_decay == 0.0。
        """
        if self.weight_decay == 0.0:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        wd = 0.0
        for layer in self._layers:
            if isinstance(layer, nn.Linear):
                wd += torch.sum(layer.weight ** 2)
        return self.weight_decay * wd

#Implements decode()
class BasicDecoder(Network):
    def __init__(self, output_activation_fct=None, weight_decay=0.0, **kwargs):
        super(BasicDecoder, self).__init__(**kwargs)
        self._output_activation_fct=output_activation_fct
        self.weight_decay = weight_decay

    def forward(self, x):
        return self.decode(x)

    def decode(self, x):
        logger.debug("Decoder::decode")
        nr_layers=len(self._layers)
        for idx,layer in enumerate(self._layers):
            if idx==nr_layers-1 and self._output_activation_fct:
                x=self._output_activation_fct(layer(x))
            else:
                x=self._activation_fct(layer(x))
        return x

    def get_weight_decay(self) -> torch.Tensor:
        if self.weight_decay == 0.0:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        wd = 0.0
        for layer in self._layers:
            if isinstance(layer, nn.Linear):
                wd += torch.sum(layer.weight ** 2)
        return self.weight_decay * wd

class SimpleEncoder(Network):
    def __init__(self,smoothing_distribution=None,num_latent_hierarchy_levels=4,**kwargs):
        super(SimpleEncoder, self).__init__(**kwargs)
        self.smoothing_distribution=smoothing_distribution

        #number of hierarchy levels in encoder. This is the number of latent
        #layers. At each hiearchy level an output layer is formed.
        self.num_latent_hierarchy_levels=4
        #number of latent units in the prior - output units for each level of
        #the hierarchy. Also number of input nodes to the decoder, first layer
        self.num_latent_units=100
        #each hierarchy has NN with num_det_layers_enc layers
        #number of deterministic units in each encoding layer. These layers map
        #input to the latent layer. 
        self.num_det_units=200
        # number of deterministic layers in each conditional p(z_i | z_{k<i})
        self.num_det_layers=2 

    def encode(self, x):
        logger.debug("encode")
        for layer in self._layers:
            if self._activation_fct:
                x=self._activation_fct(layer(x))
            else:
                x=layer(x)
        return x
    
    def hierarchical_posterior(self,x, is_training=True):
        """ This function defines a hierarchical approximate posterior distribution. The length of the output is equal 
            to num_latent_hierarchy_levels and each element in the list is a DistUtil object containing posterior distribution 
            for the group of latent units in each hierarchy level. 

        Args:
            input: a tensor containing input tensor.
            is_training: A boolean indicating whether we are building a training graph or evaluation graph.

        Returns:
            posterior: a list of DistUtil objects containing posterior parameters.
            post_samples: A list of samples from all the levels in the hierarchy, i.e. q(z_k| z_{0<i<k}, x).
        """
        logger.debug("ERROR Encoder::hierarchical_posterior")
        posterior = []
        post_samples = []
        #TODO switched off hierarchy for now.
        # import pickle
        for i in range(self.num_latent_hierarchy_levels):
            qprime=self.encode(x)
            # pickle.dump(qprime,open( "datasample.pkl", "wb" ))
            sigmoid=torch.nn.Sigmoid()
            q=sigmoid(qprime)
            #returns tensor of size n of random variables drawn from uniform

            #dist in [0,1)
            rho=torch.rand(q.size())
            posterior_dist = self.smoothing_distribution # init posterior dist.
            samples=posterior_dist.icdf(rho,q)
            posterior.append(posterior_dist)
            post_samples.append(samples)
        return posterior, post_samples
    
class SimpleDecoder(Network):
    def __init__(self,output_nodes=None,output_activation_fct=None,**kwargs):
        super(SimpleDecoder, self).__init__(**kwargs) 
        #last output layer treated separately, as it needs sigmoid activation        

        self._output_activation_fct=output_activation_fct

    def decode(self, z):
        logger.debug("Decoder::decode")
        nr_layers=len(self._layers)
        x_prime=None
        for idx,layer in enumerate(self._layers):
            if idx==nr_layers-1:
                if self._output_activation_fct:
                    x_prime=self._output_activation_fct(layer(z))
                else:
                    x_prime=self._activation_fct(layer(z))
            else:
                z=self._activation_fct(layer(z))
        return x_prime

    def decode_posterior_sample(self, zeta):
        logger.debug("Decoder::decode")  
        nr_layers=len(self._layers)
        for idx,layer in enumerate(self._layers):
            # print(idx, layer)
            if idx==nr_layers-1:

                x_prime=self._output_activation_fct(layer(zeta))
            else:
                zeta=self._activation_fct(layer(zeta))
        return x_prime

# class HierarchicalEncoder(BasicEncoder):
#     def __init__(self, 
#         activation_fct=nn.Tanh(),
#         input_dimension=784,
#         num_latent_hierarchy_levels=4,
#         num_latent_units=100,
#         num_det_units=200,
#         num_det_layers=2,
#         skip_latent_layer=False, 
#         **kwargs):
#         super(HierarchicalEncoder, self).__init__(**kwargs)
        
#         #TODO
#         #batch normalisation
#         #weight decay

#         self.num_input_units=input_dimension

#         #number of hierarchy levels in encoder. This is the number of latent
#         #layers. At each hiearchy level an output layer is formed.
#         self.num_latent_hierarchy_levels=num_latent_hierarchy_levels

#         #number of latent units in the prior - output units for each level of
#         #the hierarchy. Also number of input nodes to the decoder, first layer
#         self.num_latent_units=num_latent_units

#         #each hierarchy has NN with num_det_layers_enc layers
#         #number of deterministic units in each encoding layer. These layers map
#         #input to the latent layer. 
#         self.num_det_units=num_det_units
        
#         # number of deterministic layers in each conditional p(z_i | z_{k<i})
#         self.num_det_layers=num_det_layers

#         # for all layers except latent (output)
#         self.activation_fct=activation_fct

#         #list of all networks in the hierarchy of the encoder
#         self._networks=nn.ModuleList([])
        
#         #skip_latent_layer: instead of having a single latent layer, use
#         #Gaussian trick of VAE: construct mu+eps*sqrt(var) on each hierarchy
#         #level. This gives num_latent_hierarchy_levels latent variables, which
#         #are then combined outside this class into one layer.
#         self.skip_latent_layer=skip_latent_layer

#         self.smoothing_distribution=SpikeAndExponentialSmoother
        
#         #for each hierarchy level create a network. Input unit count will increase
#         #per level.
#         for lvl in  range(self.num_latent_hierarchy_levels):
#             network=self._create_hierarchy_network(level=lvl, skip_latent_layer=skip_latent_layer)
#             self._networks.append(network)

#     def _create_hierarchy_network(self,level=0, skip_latent_layer=False):       
#         #skip_latent_layer: instead of having a single latent layer, use
#         #Gaussian trick of VAE: construct mu+eps*sqrt(var) on each hierarchy level
#         # this is done outside this class...  
#         #TODO this should be revised with better structure for input layer config  
#         layers=[self.num_input_units+level*self.num_latent_units]+[self.num_det_units]*self.num_det_layers+[self.num_latent_units]
#         #in case we want to sample gaussian variables
#         if skip_latent_layer: 
#             layers=[self.num_input_units+level*self.num_latent_units]+[self.num_det_units]*self.num_det_layers

#         moduleLayers=nn.ModuleList([])
#         for l in range(len(layers)-1):
#             n_in_units=layers[l]
#             n_out_units=layers[l+1]

#             moduleLayers.append(nn.Linear(n_in_units,n_out_units))
#             #apply the activation function for all layers except the last
#             #(latent) layer 
#             act_fct = nn.Identity() if l==len(layers)-2 else self.activation_fct
#             moduleLayers.append(act_fct)

#         sequential=nn.Sequential(*moduleLayers)
#         return sequential

#     def hierarchical_posterior(self, in_data=None, is_training=True):
#         """ This function defines a hierarchical approximate posterior distribution. The length of the output is equal 
#             to num_latent_hierarchy_levels and each element in the list is a DistUtil object containing posterior distribution 
#             for the group of latent units in each hierarchy level. 

#         Args:
#             input: a tensor containing input tensor.
#             is_training: A boolean indicating whether we are building a training graph or evaluation graph.

#         Returns:
#             posterior: a list of DistUtil objects containing posterior parameters.
#             post_samples: A list of samples from all the levels in the hierarchy, i.e. q(z_k| z_{0<i<k}, x).
#         """
#         logger.debug("ERROR Encoder::hierarchical_posterior")
        
#         posterior = []
#         post_samples = []
#         #loop hierarchy levels. apply previously defined network to input.
#         #input is concatenation of data and latent variables per layer.
#         for lvl in range(self.num_latent_hierarchy_levels):
#             #network of hierarchy level lvl 
#             current_net=self._networks[lvl]
#             #input to this level current_net
#             current_input=torch.cat([in_data]+post_samples,dim=-1)
#             #feed network and retrieve logit
#             logit=current_net(current_input)
#             #build the posterior distribution for this hierarchy
#             #TODO make beta steerable
#             #TODO this needs a switch: training smoothing, evaluation bernoulli
#             posterior_dist = self.smoothing_distribution(logit=logit,beta=4)
#             #construct the zeta values (reparameterised logits, posterior samples)
#             samples=posterior_dist.reparameterise()
#             posterior.append(posterior_dist)
#             post_samples.append(samples)
#         return posterior, post_samples

class Decoder(BasicDecoder):
    def __init__(self,**kwargs):
        super(Decoder, self).__init__(**kwargs) 

        self._network=self._create_network()

    def _create_network(self):
        layers=self._node_sequence
        moduleLayers=nn.ModuleList([])
        
        for l in range(len(layers)):
            n_in_units=layers[l][0]
            n_out_units=layers[l][1]

            moduleLayers.append(nn.Linear(n_in_units,n_out_units))
            #apply the activation function for all layers except the last
            #(latent) layer 
            act_fct= self._output_activation_fct if l==len(layers)-1 else self._activation_fct
            moduleLayers.append(act_fct)

        sequential=nn.Sequential(*moduleLayers)
        return sequential

    def decode(self, posterior_sample):
        logger.debug("Decoder::decode")
        return self._network(posterior_sample)

if __name__=="__main__":
    logger.debug("Testing Networks")
    nw=Network()
    encoder=SimpleEncoder()
    logger.debug(encoder._layers)
    decoder=SimpleDecoder()
    logger.debug(decoder._layers)
    hierarchicalEncoder=HierarchicalEncoder()
    decoder2=Decoder()
    # prior=Prior(create_module_list=True)
    # logger.debug(prior._layers)
    logger.debug("Success")
    pass