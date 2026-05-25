# setup env w basic params
def setup_environment():
    pass

def setup_agent():
    pass


# define value function
def value_function(V_bases, m): pass

# separate reward into base dimensions
def reward_basis(basis_idx, state): pass

# separate value accordingly
def value_basis(V_bases, basis_idx): pass

# apply TD with each reward
def td_error_basis(r_i, V_i_current, V_i_next, gamma): pass


# update weights
def update_value_basis(V_bases, basis_idx, state, delta, alpha): pass

def update_motivational_weights(m, physiological_state): pass

