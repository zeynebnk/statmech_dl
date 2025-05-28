## from PMI-SYSU

from src.utils import *


def gen_field(P, D, N, λ0, device):

    w = torch.normal(0, 1, size=(P, 1, D)).to(device)
    x = torch.normal(0, 1, size=(P, D, N)).to(device)
    xt = torch.normal(0, 1, size=(P, D, 1)).to(device)

    y = w @ x / np.sqrt(D)    
    yt = w @ xt / np.sqrt(D)  

    c1 = x @ y.permute(0, 2, 1) / N  
    s1 = xt @ c1.permute(0, 2, 1)    
    h1 = yt * s1                     
    h1 = h1.mean(dim=0)              

    c2 = y.squeeze(1).pow(2).sum(dim=1, keepdim=True) / N  
    s2 = xt.squeeze(-1) * c2                              
    h2 = yt.squeeze(-1) * s2                              
    h2 = h2.mean(dim=0).unsqueeze(0)                      

    h_zero = torch.zeros((D+1, 1)).to(device)
    h = torch.cat((h1, h2), dim=0)
    h = torch.cat((h, h_zero), dim=1)

    s2 = s2.unsqueeze(1)
    s_zero = torch.zeros((P, D+1, 1)).to(device)
    s = torch.cat((s1, s2), dim=1)
    s = torch.cat((s, s_zero), dim=2)

    J = - s.reshape(P, -1, 1) @ s.reshape(P, 1, -1)
    J = J.mean(dim=0)

    diag_J = torch.diagonal(J)
    λ = λ0 - diag_J

    return J, h, λ


def AMP_iteration(J, h, λ, β, θ, steps, device, showlog):

    D2 = J.shape[0]
    m_list = torch.normal(0,1,size=(D2,)).to(device)
    v_list = torch.normal(0,1,size=(D2,)).to(device)

    final_error = 1
    iters = 0

    while final_error > 1e-6 and iters < steps:

        new_m_list = torch.zeros(D2).to(device)
        new_v_list = torch.zeros(D2).to(device)

        for i in range(D2):
            M_hat = J[i,:] * m_list
            M_hat[i] = 0
            M_i = β * M_hat.sum()

            V_hat = J[i,:].pow(2) * v_list
            V_hat[i] = 0
            V_i = β ** 2 * V_hat.sum()

            new_m_list[i] = (β * h[i] + M_i) / (β * λ[i] - V_i)
            new_v_list[i] = 1 / (β * λ[i] - V_i)

        ep_m = MSE(new_m_list, m_list)
        ep_v = MSE(new_v_list, v_list)

        if showlog:
            print(f"|iter| {iters+1} |error| m = {ep_m.item():.8f}, v = {ep_v.item():.8f}")

        m_list = θ * m_list + (1-θ) * new_m_list
        v_list = θ * v_list + (1-θ) * new_v_list

        final_error = (ep_m + ep_v).cpu().item()
        iters += 1

    return m_list, v_list, iters, final_error


def AMP_test(weight, te_X, te_y):
    N = te_X.shape[2]
    attn = te_X @ te_X.permute(0, 2, 1) @ weight @ te_X / N
    pre = attn[:, -1, -1]

    loss = MSE(pre, te_y).cpu().item()

    return loss


def calculate_contrast_ratio(matrix):
    identity_matrix = np.eye(matrix.shape[0], matrix.shape[1])

    dot_product = np.sum(matrix * identity_matrix)
    norm_matrix = np.linalg.norm(matrix)
    norm_identity = np.linalg.norm(identity_matrix)
    contrast_ratio = dot_product / (norm_matrix * norm_identity)

    return contrast_ratio


def energy_landscape(P, D, N, λ0, mc_steps, dev="cpu"):

    device = torch.device(dev)
    tsne = TSNE()
    
    J, h, λ = gen_field(P, D, N, λ0, device)
    h = h.reshape(-1)

    print("Field generation done")

    weights = torch.normal(0, 1, (mc_steps, (D+1)*(D+1))).to(device)
    energies = []

    for i in tqdm(range(mc_steps)):
        W = weights[i]
        E = - 0.5 * W.reshape(1, -1) @ J @ W.reshape(-1, 1) - W.reshape(1, -1) @ h + 0.5 * W.reshape(1, -1).pow(2) @ λ.reshape(-1, 1)
        energies.append(E.cpu().item())

    print("Energy calculation done")

    tsne_weights = tsne.fit_transform(weights.cpu().numpy())

    print("TSNE done")

    min_x = int(min(tsne_weights[:, 0]))
    max_x = int(max(tsne_weights[:, 0]))
    min_y = int(min(tsne_weights[:, 1]))
    max_y = int(max(tsne_weights[:, 1]))

    N_sparse = 200
    x_sparse, y_sparse = np.meshgrid(np.linspace(min_x, max_x, N_sparse), np.linspace(min_y, max_y, N_sparse))

    z_list = np.array([x_sparse.reshape(-1), y_sparse.reshape(-1)]).transpose()
    func = RBFInterpolator(tsne_weights, energies, neighbors=100, smoothing=50, kernel='linear')
    z_result = func(z_list)

    z_sparse = z_result.reshape(N_sparse, N_sparse)

    print("Interpolation done")

    return x_sparse, y_sparse, z_sparse
