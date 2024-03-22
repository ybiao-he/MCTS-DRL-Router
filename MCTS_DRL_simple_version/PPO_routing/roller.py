import numpy as np
from models import GaussianModel


class Roller:

    def __init__(self, env, model, num_steps, gamma= 0.99, lam= 0.97):

        self.env = env
        self.model = model
        self.num_steps = num_steps
        self.gamma = gamma
        self.lam = lam
        self._obs_vec = None
        self._obs_vis = None
        self._rews = None
        self._dones = None
        # self._first_reset = False

    def reset(self):
        self._obs_vec, self._obs_vis, self._rews, self._dones = self.env.reset()

    def rollout(self, epoch):

        self.reset()
        
        vec_obses = []
        vis_obses = []
        rews = []
        dones = []
        actions = []
        values = []
        logp = []
        ep_rews = []
        ep_lens = []


        ep_len = 0
        ep_rew = 0
        for step in range(self.num_steps):              # Run each env for num_steps and gather trajectory rollouts

            if self._dones:

                actions_t, logp_t, values_t = self.model.get_action_logp_value({"vec_obs": self._obs_vec, "vis_obs": self._obs_vis})
                vec_obses.append(self._obs_vec) if self.env.env_info.is_vector else None
                vis_obses.append(self._obs_vis) if self.env.env_info.is_visual else None
                
                actions.append(actions_t)
                values.append(values_t)
                logp.append(logp_t)

                rews.append(0)
                dones.append(self._dones)

                self.reset()
                ep_lens.append(ep_len)
                ep_rews.append(ep_rew)
                ep_len = 0
                ep_rew = 0
                continue

            actions_t, logp_t, values_t = self.model.get_action_logp_value({"vec_obs": self._obs_vec, "vis_obs": self._obs_vis})
            # print(self.model.p_all({"vec_obs": self._obs_vec, "vis_obs": self._obs_vis}))

            vec_obses.append(self._obs_vec) if self.env.env_info.is_vector else None
            vis_obses.append(self._obs_vis) if self.env.env_info.is_visual else None
            
            actions.append(actions_t)
            values.append(values_t)
            logp.append(logp_t)

            self._obs_vec, self._obs_vis, self._rews, self._dones, infos = self.env.step(actions_t)

            rews.append(self._rews)
            dones.append(False)
            
            ep_len += 1
            ep_rew += self._rews

        ep_lens.append(ep_len)
        ep_rews.append(ep_rew)
        print(len(ep_lens), np.mean(ep_rews))
        # print(ep_rews)

        # if not self._dones:
        #     ep_lens.append(ep_len)
        #     ep_rews.append(ep_rew)
                
        """
            End of for loop
            ---------------
            Get last Values for BOOTSTRAPING
        """
        actions_t, logp_t, values_t = self.model.get_action_logp_value({"vec_obs": self._obs_vec, "vis_obs": self._obs_vis})
        last_values = values_t                          # Bootstraping

        vec_obses = np.array(vec_obses, dtype= np.float32) if self.env.env_info.is_vector else None
        vis_obses = np.array(vis_obses, dtype= np.float32) if self.env.env_info.is_visual else None
        rews = np.array(rews, dtype=np.float32)
        dones = np.array(dones, dtype= np.bool)
        values = np.array(values, dtype=np.float32)
        logp = np.array(logp, dtype=np.float32)
        actions = np.array(actions, dtype=np.float32) if isinstance(self.model, GaussianModel) else np.array(actions, dtype=np.int32)
        
        """
            Discount / Bootstrap Values and calc Advantages
            -----------------------------------------------
        """
        returns = np.zeros_like(rews)
        advs = np.zeros_like(rews)
        last_gae_lam = 0

        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_non_terminal = 1.0 - self._dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - dones[t + 1]
                next_values = values[t + 1]
            
            delta = rews[t] + self.gamma * next_values * next_non_terminal - values[t]
            advs[t] = last_gae_lam = delta + self.gamma * self.lam * next_non_terminal * last_gae_lam
            
        returns = advs + values                         # ADV = RETURNS - VALUES
        advs = (advs - advs.mean()) / (advs.std())      # Normalize ADVs

        return self._flattened_rollout(vec_obses, vis_obses, rews, dones, actions, logp, values, advs, returns), \
               {'ep_rews': ep_rews, 'ep_lens': ep_lens}

    def _flattened_rollout(self, vec_obses, vis_obses, rews, dones, actions, logp, values, advs, returns):

        if self.env.env_info.is_visual:         # Reshape visual obs --> flatten array
            d1, d2, d3, d4, d5 = vis_obses.shape
            vis_obses = vis_obses.reshape(d1 * d2, d3, d4, d5)
        
        if self.env.env_info.is_vector:         # Reshape flatten vec obs
            vec_obses = vec_obses.reshape(-1, vec_obses.shape[-1])
        
        if isinstance(self.model, GaussianModel):
            actions = actions.reshape(-1, actions.shape[-1])
        else:
            actions = actions.flatten()

        return{     'vec_obses': vec_obses, 
                    'vis_obses': vis_obses,
                    'rews': rews.flatten(), 
                    'dones': dones.flatten(), 
                    'actions': actions,
                    'logp': logp.flatten(), 
                    'values': values.flatten(), 
                    'advs': advs.flatten(), 
                    'returns': returns.flatten()}
