import torch.nn as nn

def set_fr_module(args):
    """
    Create a frequency-representation module
    """
    net = None
    if args.fr_module_type == 'fr':
        assert args.fr_size == args.fr_inner_dim * args.fr_upsampling, \
            'The desired size of the frequency representation (fr_size) must be equal to inner_dim*upsampling'
        net = FrequencyRepresentationModule(signal_dim=args.signal_dim, n_filters=args.fr_n_filters,
                                            inner_dim=args.fr_inner_dim, n_layers=args.fr_n_layers,
                                            upsampling=args.fr_upsampling, kernel_size=args.fr_kernel_size,
                                            kernel_out=args.fr_kernel_out)
    else:
        raise NotImplementedError('Frequency representation module type not implemented')

    return net

class FrequencyRepresentationModule(nn.Module):
    def __init__(self, signal_dim=50, n_filters=8, n_layers=3, inner_dim=125,
                 kernel_size=3, upsampling=8, kernel_out=25):
        super().__init__()
        self.fr_size = inner_dim * upsampling
        self.n_filters = n_filters
        self.in_layer = nn.Linear(2 * signal_dim, inner_dim * n_filters, bias=False)
        mod = []
        for n in range(n_layers):
            mod += [
                nn.Conv1d(n_filters, n_filters, kernel_size=kernel_size, padding=kernel_size // 2, bias=False,
                          padding_mode='circular'),
                nn.BatchNorm1d(n_filters),
                nn.ReLU(),
            ]
        self.mod = nn.Sequential(*mod)
        self.out_layer = nn.ConvTranspose1d(n_filters, 1, kernel_out, stride=upsampling,
                                            padding=(kernel_out - upsampling + 1) // 2, output_padding=1, bias=False)

    def forward(self, inp):
        bsz = inp.size(0)
        inp = inp.view(bsz, -1)
        x = self.in_layer(inp).view(bsz, self.n_filters, -1)
        x = self.mod(x)
        x = self.out_layer(x).view(bsz, -1)
        return x