import argparse


def _parse_bool(value):
    if isinstance(value, bool):
        return value

    lowered = value.strip().lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False

    raise argparse.ArgumentTypeError(f"Expected a boolean value, got '{value}'.")


def _positive_int(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError(
            f"Expected a positive integer, got '{value}'."
        )
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(description='LEAP')
    parser.add_argument('--seed', type=int, default=0,help='Seed - Test your luck!')   
    parser.add_argument('--intensity', type=_parse_bool, default=True,help='Intensity')
    parser.add_argument('--model', type=str, default='ncde',help='Model Name')
    parser.add_argument('--h_channels', type=int, default=49,help='Hidden Channels')   
    parser.add_argument('--ode_hidden_hidden_channels', type=int, default=40,help='ODE Func Hidden Hidden Channels')    
    parser.add_argument('--hh_channels', type=int, default=49,help='Hidden Hidden Channels')          
    parser.add_argument('--layers', type=int, default=4,help='Num of Hidden Layers')   
    parser.add_argument('--lr', type=float, default=0.0001,help='Learning Rate')  
    parser.add_argument('--epoch',type=_positive_int,default = 200,help ='Epoch') 
    parser.add_argument('--step_mode', type=str, default='valloss',help='Learning Rate Scheduler')
    parser.add_argument('--dataset_name', type=str, choices=['mujoco', 'physionet', 'crypto'], default='mujoco', help='Dataset Name')
    parser.add_argument('--batch_size', type=_positive_int, default=1024, help='Training/evaluation batch size')
    parser.add_argument('--missing_rate', type=float, default=0.3,help='Missing Rate')
    parser.add_argument('--method', type=str, default='rk4', help='solver method')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight_decay')
    parser.add_argument('--loss', type=str, default='mse', help='loss setting')
    parser.add_argument('--reg', type=str, default='l2', help='regularization setting')
    parser.add_argument('--scale', type=float, default=0.01, help='regularization setting')
    parser.add_argument('--time_seq', type=int, default=50, help='time_seq')
    parser.add_argument('--y_seq', type=int, default=10, help='y_seq')
    parser.add_argument('--gpu', type=int, default=0,help='GPU')
    parser.add_argument('--c1', type=float, default=0.0, help='Auxiliary ODE loss coefficient')
    parser.add_argument('--c2', type=float, default=0.0, help='Auxiliary ODE loss coefficient')
    parser.add_argument('--ode_model', type=str, default='ncde_forecasting', help='ODE model for ODEvsSDE.py')
    parser.add_argument('--mc_samples', type=int, default=20, help='SDE Monte Carlo paths at final test')
    parser.add_argument('--mc_seed', type=int, default=12345, help='Monte Carlo seed')
    parser.add_argument('--explosion_threshold', type=float, default=100.0, help='Absolute SDE explosion threshold')
    parser.add_argument(
        '--sde_input_option',
        type=int,
        choices=range(7),
        default=None,
        help='Drift/input option for --model diffusionsde (0-6)',
    )
    parser.add_argument(
        '--sde_noise_option',
        type=int,
        choices=range(25),
        default=None,
        help='Diffusion function for --model diffusionsde (0-24)',
    )
    parser.add_argument(
        '--sde_mixture_options',
        type=int,
        nargs=3,
        default=None,
        metavar=('G1', 'G2', 'G3'),
        help=(
            'Three catalogue options (1-23) for the mixture diffusion '
            '(noise_option=24). Default: 16 23 6 '
            '(mlp_time / linear_time_plus_linear_state / diagonal_state).'
        ),
    )
    parser.add_argument(
    "--test_missing_rates",
    type=float,
    nargs="+",
    default=[0.3, 0.5, 0.7, 0.9],
    )

    parser.add_argument(
        "--input_noise_levels",
        type=float,
        nargs="+",
        default=[0.0, 0.05, 0.1, 0.2],
    )

    parser.add_argument(
        "--corruption_repeats",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--corruption_seed",
        type=int,
        default=24680,
    )

    parser.add_argument(
        "--missing_pattern",
        choices=["random", "block", "tail"],
        default="random",
    )
    return parser.parse_args()
