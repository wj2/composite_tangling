
import argparse
import itertools as it
import pickle

import composite_tangling.code_analysis as ca

def create_parser():
    parser = argparse.ArgumentParser(description='fit several autoencoders')
    parser.add_argument('-o', '--output_path', default='ct_results.pkl',
                        type=str, help='folder to save the output in')
    parser.add_argument('snr', nargs='+', type=float, help='SNRs to use')
    parser.add_argument('--n_feats', nargs='+', type=int, default=(2,),
                        help='numbers of features to use')
    parser.add_argument('--n_trades', type=int, default=20,
                        help='numbers of tradeoffs to use')
    parser.add_argument('--n_values', nargs='+', type=int, default=(2,),
                        help='numbers of values for each feature to use')
    parser.add_argument('--n_neurons', nargs='+', type=int, default=(150,),
                        help='numbers of neurons in the population')
    parser.add_argument('--trade_reps', type=int, default=10,
                        help='numbers repetitions across codes')
    parser.add_argument('--n_reps', type=int, default=2,
                        help='number of repetitions within codes')
    parser.add_argument('--no_shattering', action='store_false', default=True)
    parser.add_argument('--no_discrim', action='store_false', default=True)
    parser.add_argument('--no_mse', action='store_false', default=True)
    parser.add_argument('--eps', default=0, type=float,
                        help='offset from edges of full tradeoff')
    parser.add_argument('--training_noise', default=False, action='store_true',
                        help='use noise in training examples')
    return parser

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    snrs = args.snr
    noise_var = 1
    n_trades = args.n_trades
    n_feats = args.n_feats
    n_values = args.n_values
    n_neurs = args.n_neurons
    trade_reps = args.trade_reps
    n_reps = args.n_reps
    eps = args.eps
    tn = args.training_noise

    compute_kwargs = {'compute_shattering':args.no_shattering,
                      'compute_discrim':args.no_discrim,
                      'compute_mse':args.no_mse}
    trades, full_dict = ca.get_multiple_tradeoffs(snrs, n_trades, n_feats,
                                                  n_values, n_neurs,
                                                  trade_reps=trade_reps,
                                                  n_reps=n_reps, eps=eps,
                                                  train_noise=tn,
                                                  **compute_kwargs)
    
    pickle.dump((args, trades, full_dict), open(args.output_path, 'wb'))
