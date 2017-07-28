from a_nice_mc.utils.nice import NiceLayer, NiceNetwork


def create_nice_network(x_dim, v_dim, args):
    net = NiceNetwork(x_dim, v_dim)
    for dims, name, swap in args:
        net.append(NiceLayer(dims, name, swap))
    return net
