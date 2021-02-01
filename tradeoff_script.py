
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

    arg_combs = it.product(snrs, n_feats, n_values, n_neurs)
    full_dict = {}
    for (snr, n_feat, n_val, n_neur) in arg_combs:
        out = ca.get_ccgp_dim_tradeoff(snr, n_trades, n_feat, n_val,
                                       n_neur, trade_reps=trade_reps,
                                       n_reps=n_reps, train_noise=tn,
                                       noise_var=noise_var, eps=eps,
                                       **compute_kwargs)
        trades, metrics, info = out
        ids = info['min_dist_code']
        p_l, p_n = ca.get_lin_nonlin_pwr([snr], trades, noise_var=noise_var)
        
        errs, t1, t2 = ca.ccgp_error_rate(ids[:, 0, 0], ids[:, 0, 1], p_l, p_n,
                                          noise_var, n_neur, n=10000)
        shatt_err = ca.partition_error_rate(p_n, noise_var, n_val**n_feat)

        theor = {'ccgp':1 - errs, 'shattering':1 - shatt_err}

        out_dict = dict(metrics=metrics, info=info, theory=theor)
        full_dict[(snr, n_feat, n_val)] = out_dict
    
    pickle.dump((args, trades, full_dict), open(args.output_path, 'wb'))
