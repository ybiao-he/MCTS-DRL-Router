import tensorflow as tf
import numpy as np
from wrapper_env import EnvInfo

EPS = 1e-8      # Const to avoid Machine Precission Error


class CategoricalModel(tf.keras.Model):
    """
        Categorical Model for Discrete Action Spaces
        --------------------------------------------

            Input:

                Network with foward pass and EnvInfo Object

            Returns:   

                call:                   logits, values from Neural Network 
                                        with defined forward pass

                get-action-lop-value:   logp, action, value 
                                        (action drawn from Random Categ. Dist)

                logp:                   Log probability for action x

                entropy:                Entropy Term from logits

    """
    def __init__(self, network= None, env_info= EnvInfo):
        super().__init__('CategoricalModel')

        self.env_info = env_info
        self.act_size = self.env_info.act_size                              # Number of possible actions
        self.forward = network['forward']                                   # Get feed forward chain
        self.all_networks = network['trainable_networks']                   # Get all trainable networks
  
    def pd(self, logits, mask=None):
        if mask is not None:
            logp_all = tf.nn.log_softmax(logits)
            p = tf.math.exp(logp_all)[0]
            new_dist = p.numpy()*mask
            # print(new_dist, p, logits)
            if sum(new_dist)==0:
                new_dist = np.ones(len(new_dist))
            new_dist = new_dist/sum(new_dist)
            # print(mask, p, new_dist)
            return np.random.choice(range(len(new_dist)), 1, p = new_dist)
        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)        # Draw from Random Categorical Distribution

    def predict(self, inputs):
        return self.forward(inp = inputs)
        
    def get_action_logp_value(self, obs, mask=None):
        """
            Returns:

            logits --> Last layer of Neural Network without activation function \n
            logp --> SOFTMAX of logits which squashes logits between 0 .. 1 and returns log probabilities \n
            actions --> drawn from normal distribution
        """
        logits, values = self.predict(obs)                                  # Returns np arrays on predict | Input: np array or tensor or list
        actions = self.pd(logits, mask=mask)
        logp_t = self.logp(logits, actions) 
        return np.squeeze(actions), np.squeeze(logp_t), np.squeeze(values) 

    def logp(self, logits, action):
        """
            Returns:
            
            logp based on the action drawn from prob-distribution \n
            indexes in the logp_all with one_hot
        """
        logp_all = tf.nn.log_softmax(logits)
        one_hot = tf.one_hot(action, depth= self.act_size)
        logp = tf.reduce_sum( one_hot * logp_all, axis= -1)
        return logp

    def p_all(self, obs):
        logits, values = self.predict(obs)
        logp_all = tf.nn.log_softmax(logits)
        return tf.math.exp(logp_all)

    def entropy(self, logits= None):
        """
            Entropy term for more randomness which means more exploration \n
            Based on OpenAI Baselines implementation
        """
        a0 = logits - tf.reduce_max(logits, axis= -1, keepdims=True)
        exp_a0 = tf.exp(a0)
        z0 = tf.reduce_sum(exp_a0, axis= -1, keepdims=True)
        p0 = exp_a0 / z0
        entropy = tf.reduce_sum(p0 * (tf.math.log(z0) - a0), axis= -1)
        return entropy
