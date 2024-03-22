import tensorflow as tf
import numpy as np
from wrapper_env import EnvInfo

mapping = {}


def register(name):
    def _thunk(func):
        mapping[name] = func
        return func
    return _thunk


def mlp(hidden_sizes= (32, 32), output_size= 1, activation= 'relu', activation_output= None, kernel_initalizer= 'glorot_uniform', name= 'MLP'):
    """
        MLP - Multilayer Perceptron
        ---------------------------

            Hidden Sizes = [32,32] Size of HIDDEN Layers 
            Output Size = (1) Size of OUTPUT Layer
            Activation = RELU
            Output Activation  = NONE
            Kernel Initializer = glorot uniform
            bias inintializer =  ZEROS

    """
    model = tf.keras.Sequential(name= name)
    layer_idx = 1
    
    for h in hidden_sizes:
        model.add(tf.keras.layers.Dense(units= h, activation= activation, name= name+str(layer_idx), kernel_initializer= kernel_initalizer, bias_initializer= 'zeros'))
        layer_idx += 1
    
    model.add(tf.keras.layers.Dense(units= output_size, activation= activation_output, name= name + '_output'))

    return model

# need to be revised
@register("simple_actor_critic")
def simple_actor_critic(hidden_sizes=(32, 32), activation='relu', activation_output=None, 
                        kernel_initalizer='glorot_uniform', name='simple_actor_critic', env_info=EnvInfo):

    info_dim = [6,4,4,2]
    embed_dim = [6,8,8,4]

    actor_embed_input, actor_embed_output = info_embedding(info_dim=info_dim, embed_dim=embed_dim, name='actor')
    critic_embed_input, critic_embed_output = info_embedding(info_dim=info_dim, embed_dim=embed_dim, name='critic')

    actor = mlp(hidden_sizes=hidden_sizes, output_size=env_info.act_size, activation=activation, 
                 activation_output=activation_output, name="actor", kernel_initalizer=kernel_initalizer)
    
    critic = mlp(hidden_sizes= hidden_sizes, output_size= 1, activation= activation, 
                  activation_output= activation_output, name="actor", kernel_initalizer= kernel_initalizer)

    print('Model Summary: ' + name)

    # actor.build(input_shape = (None, sum(embed_dim)))

    # critic.build(input_shape = (None, sum(embed_dim)))

    actor.build(input_shape = (None, sum([6])))

    critic.build(input_shape = (None, sum([6])))

    actor_out = actor(actor_embed_output)
    critic_out = critic(critic_embed_output)

    _actor = tf.keras.Model(actor_embed_input, actor_out, name=name)
    _critic = tf.keras.Model(critic_embed_input, critic_out, name=name)

    _actor.summary()
    _critic.summary()

    def forward(inp= None):
        logits = _actor(inp['vec_obs'])
        values = _critic(inp['vec_obs'])
        return logits, values

    return {"forward": forward, "trainable_networks": [_actor, _critic]}


def info_embedding(info_dim=[6, 4, 4, 2], embed_dim=[6,8,8,4], name="actor"):
    """
        info_dim contains feature dimensions of current net (6), other net (4), paths (4) and obstacles (2)
        embed_dim contains embeded dimensions of current net, other net, paths and obstacles
    """
    current_nets = tf.keras.Input(shape=(info_dim[0],), name=name+"_current_nets")

    other_net_embed_dim = embed_dim[1]
    other_net_name = name+"_other_net"
    model_other_net = mlp(hidden_sizes=[], output_size=other_net_embed_dim, activation='relu', 
                activation_output=None, name=other_net_name, kernel_initalizer='glorot_uniform')

    model_other_net.build(input_shape=(None,None,info_dim[1]))
    model_other_net.summary()

    paths_embed_dim = embed_dim[2]
    paths_name = name+"_paths"
    model_paths = mlp(hidden_sizes=[], output_size=paths_embed_dim, activation='relu', 
                activation_output=None, name=paths_name, kernel_initalizer='glorot_uniform')

    model_paths.build(input_shape=(None,None,info_dim[2]))
    model_paths.summary()

    obstacles_embed_dim = embed_dim[3]
    obstacles_name = name+"_obstacles"
    model_obstacles = mlp(hidden_sizes=[], output_size=obstacles_embed_dim, activation='relu', 
                activation_output=None, name=obstacles_name, kernel_initalizer='glorot_uniform')

    model_obstacles.build(input_shape=(None,None,info_dim[3]))
    model_obstacles.summary()

    # aggregate_outputs = tf.keras.layers.Concatenate()([current_nets,tf.math.reduce_mean(model_other_net.output, 1),
    #                                                   tf.math.reduce_mean(model_paths.output,1),
    #                                                   tf.math.reduce_mean(model_obstacles.output,1)])

    # aggregate_inputs = [current_nets, model_other_net.input, model_paths.input, model_obstacles.input]


    aggregate_outputs = tf.keras.layers.Concatenate()([current_nets])

    aggregate_inputs = [current_nets]

    return aggregate_inputs, aggregate_outputs

def network_builder(name):
    """
        If you want to register your own network outside models.py.

        Usage Example:
        -------------

        import register

        @register("your_network_name")

        def your_network_define(**net_kwargs): return network_fn
    """
    if callable(name):
        return name
    elif name in mapping:
        return mapping[name]
    else:
        raise ValueError('Unknown network type: {}'.format(name))