import torch

def estimate_gradient_rge(z, classifier, decoder, original_pre, criterion, loss_0, mu=0.005, q=192):
    """
    Randomized Gradient Estimation (RGE)
    Pokes the model in 'q' random directions to estimate the gradient.
    """
    batch_size = z.size(0)
    channel = z.size(1)
    h = z.size(2)
    w = z.size(3)
    d = channel * h * w  # Total number of dimensions in the latent space

    # Flatten the latent representation so we can do math on it
    z_flat = torch.flatten(z, start_dim=1).detach()
    grad_est = torch.zeros(batch_size, d).cuda()

    # Poking Loop
    with torch.no_grad():
        for _ in range(q):
            # 1. Generate a random direction vector (u)
            u = torch.normal(0, 100, size=(batch_size, d))
            u_norm = torch.norm(u, p=2, dim=1).reshape(batch_size, 1).expand(batch_size, d)
            u = torch.div(u, u_norm).cuda()  # Normalize it to have a length of 1

            # 2. Apply the "wiggle" to the latent vector
            z_wiggled = z_flat + mu * u
            z_wiggled = z_wiggled.view(batch_size, channel, h, w)

            # 3. Query the Black Box
            wiggled_pre = classifier(decoder(z_wiggled))
            
            # 4. Calculate how the loss changed
            loss_tmp = criterion(wiggled_pre, original_pre)
            loss_diff = (loss_tmp - loss_0).detach() # Make sure it's detached from the graph

            # 5. Accumulate the gradient estimate
            grad_est = grad_est + (d / q) * u * loss_diff.view(-1, 1).expand_as(u) / mu

    return grad_est.detach()


def estimate_gradient_cge(z, classifier, decoder, original_pre, criterion, mu=0.005):
    """
    Coordinatewise Gradient Estimation (CGE)
    Pokes the model exactly once for every single dimension in the latent space.
    """
    batch_size = z.size(0)
    channel = z.size(1)
    h = z.size(2)
    w = z.size(3)
    d = channel * h * w

    z_flat = torch.flatten(z, start_dim=1).detach()
    grad_est = torch.zeros(batch_size, d).cuda()

    # Poke every single dimension one by one
    with torch.no_grad():
        for k in range(d):
            u = torch.zeros(batch_size, d).cuda()
            u[:, k] = 1  # Isolate dimension 'k'

            # Poke forward and backward
            z_plus = z_flat + mu * u
            z_minus = z_flat - mu * u

            z_plus = z_plus.view(batch_size, channel, h, w)
            z_minus = z_minus.view(batch_size, channel, h, w)

            # Query Black Box for both
            pre_plus = classifier(decoder(z_plus))
            pre_minus = classifier(decoder(z_minus))

            loss_plus = criterion(pre_plus, original_pre)
            loss_minus = criterion(pre_minus, original_pre)

            # Calculate exact difference
            loss_diff = (loss_plus - loss_minus).detach()
            grad_est = grad_est + u * loss_diff.view(-1, 1).expand_as(u) / (2 * mu)

    return grad_est.detach()