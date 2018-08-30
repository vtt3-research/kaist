import torch

# DA layer for the train network
def autodial2d(input, alpha, batch_size = 256, split = 1):
    if input is not None and input.dim() != 2:
        raise ValueError("Expected 4D tensor as input, got {}D tensor instead.".format(input.dim()))

    source_input = input.narrow(0, 0, split)
    target_input = input.narrow(0, split, batch_size - split)
    eps = 1e-5

    mu_st = (alpha * source_input.mean(0, True)) / split + (1 - alpha) * (target_input.mean(0, True)) / (batch_size - split)
    mu_ts = (1 - alpha) * (torch.mean(source_input, 0, True)) / split + alpha * (torch.mean(target_input, 0, True)) / (batch_size - split)
    dev_st = alpha * torch.div(torch.sum((source_input-mu_st.expand_as(source_input).mean(0)).pow(2), 0), split) + \
             (1 - alpha) * torch.div(torch.sum((target_input-mu_ts.expand_as(target_input).mean(0)).pow(2), 0), batch_size - split)
    dev_ts = (1 - alpha) * torch.div(torch.sum((source_input - mu_ts.expand_as(source_input).mean(0)).pow(2), 0), split) + \
             alpha * torch.div(torch.sum((target_input - mu_st.expand_as(target_input).mean(0)).pow(2), 0), batch_size - split)

    source_output = (source_input - mu_st.expand_as(source_input)) / ((dev_st + eps).pow(0.5)).expand_as(source_input)
    target_output = (target_input - mu_ts.expand_as(target_input)) / ((dev_ts + eps).pow(0.5)).expand_as(target_input)

    return torch.cat((source_output, target_output),0)

# batch normalization layer instead of DA layer for test
def batchnorm2d(input):
    if input is not None and input.dim() != 2:
        raise ValueError("Expected 4D tensor as input, got {}D tensor instead.".format(input.dim()))

    eps = 1e-5
    batch_size = input.size(0)
    mu = input.mean(0, True)
    dev = ((input - mu.expand_as(input)).pow(2)).mean(0, True) / batch_size
    output = (input - mu.expand_as(input))/ ((dev + eps).pow(0.5)).expand_as(input)

    return output

# gradient formula for backpropagation in training
def autodial_grad2d(input, alpha, batch_size, split, grad_output):
    source_input = torch.Tensor.narrow(input, 0, 0, split)
    target_input = torch.Tensor.narrow(input, 0, split, batch_size - split)
    source_grad_output = torch.Tensor.narrow(grad_output, 0, 0, split)
    target_grad_output = torch.Tensor.narrow(grad_output, 0, split, batch_size - split)
    eps = 1e-5

    mu_st = (alpha * source_input.mean(0, True)) / split + (1 - alpha) * (target_input.mean(0, True)) / (
    batch_size - split)
    mu_ts = (1 - alpha) * (torch.mean(source_input, 0, True)) / split + alpha * (torch.mean(target_input, 0, True)) / (
    batch_size - split)
    dev_st = alpha * torch.div(torch.sum((source_input - mu_st.expand_as(source_input).mean(0)).pow(2), 0), split) + \
             (1 - alpha) * torch.div(torch.sum((target_input - mu_ts.expand_as(target_input).mean(0)).pow(2), 0), batch_size - split)
    dev_ts = (1 - alpha) * torch.div(torch.sum((source_input - mu_ts.expand_as(source_input).mean(0)).pow(2), 0), split) + \
             alpha * torch.div(torch.sum((target_input - mu_st.expand_as(target_input).mean(0)).pow(2), 0), batch_size - split)


    y_st = (source_input - mu_ts.expand_as(source_input)) / ((dev_ts + eps).pow(0.5)).expand_as(source_input)
    y_ts = (target_input - mu_ts.expand_as(target_input)) / ((dev_st + eps).pow(0.5)).expand_as(target_input)
    source_grad_input = 1 / (eps  + dev_st).pow(0.5) * (source_grad_output - alpha / split * (source_grad_output.sum(0, True) + torch.mul(source_input, (torch.mul(source_input, source_grad_output).sum(0, True).expand_as(source_input)))) - \
                        1 / (eps + dev_ts).pow(0.5) * (1 - alpha) / (batch_size - split) * (target_grad_output.sum(0, True) + torch.mul(y_st, (torch.mul(source_input, target_grad_output).sum(0, True)).expand_as(y_st))))
    target_grad_input = 1 / (eps + dev_ts).pow(0.5) * (target_grad_output - alpha / (batch_size - split) * (target_grad_output.sum(0, True) + torch.mul(target_input, (torch.mul(target_input, target_grad_output).sum(0, True)).expand_as(target_input)))) - \
                        1 / (eps + dev_st).pow(0.5) * (1 - alpha) / split * (source_grad_output.sum(0, True) + torch.mul(y_ts, (torch.mul(source_input, source_grad_output).sum(0, True)).expand_as(source_input)))
    grad_input = torch.cat((source_grad_input, target_grad_input),0)
    return grad_input

def autodial_new(input, alpha, prob, batch_size, split, grad_output):
    source_input = torch.Tensor.narrow(input, 0, 0, split)
    target_input = torch.Tensor.narrow(input, 0, split, batch_size - split)
    source_grad_output = torch.Tensor.narrow(grad_output, 0, 0, split)
    target_grad_output = torch.Tensor.narrow(grad_output, 0, split, batch_size - split)
    eps = 1e-5

    #mu_st = (alpha * )


    return 0