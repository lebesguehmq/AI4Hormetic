import os
import random
import time
import tensorflow as tf
import numpy as np
import pandas
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import scipy.io
from scipy.interpolate import griddata
from pyDOE import lhs

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  
start_time = time.time()

# 
N_BOOTSTRAP = 50
BOOTSTRAP_SEEDS = list(range(41, N_BOOTSTRAP + 1))
MODEL_INITIALIZATION_SEED = 42


def set_random_seed(seed):
     
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

with tf.device('/cpu:0'):
    class PhysicsInformedNN:
        def __init__(self, Ds_train, Rs_train, x_f, d_f, lb, ub, layers, layers_h, sf):


            self.sf = sf

            # Data for training
            self.Ds_train = Ds_train
            self.Rs_train = Rs_train
            self.x_f = x_f
            self.d_f = d_f

            # Time division s
            # self.M = len(t_f) - 1
            # self.tau = t_f[1] - t_f[0]

            # bounds
            self.lb = lb
            self.ub = ub

            # initailize NN
            self.weights, self.biases = self.initialize_NN(layers)
            self.weights_h, self.biases_h = self.initialize_NN(layers_h)

            # parameters
            bound_rho = [0.01, 8]
            bound_lamda = [1, 35]
            bound_theta = [0.01, 0.2]

            self.rho = bound_rho[0] + (bound_rho[1] - bound_rho[0]) * tf.sigmoid(
                tf.Variable([0.05], dtype=tf.float32, trainable=True))
            self.lamda = bound_lamda[0] + (bound_lamda[1] - bound_lamda[0]) * tf.sigmoid(
                tf.Variable([0.1], dtype=tf.float32, trainable=True))
            self.theta = bound_theta[0] + (bound_theta[1] - bound_theta[0]) * tf.sigmoid(
                tf.Variable([0.15], dtype=tf.float32, trainable=True))


            # tf placeholders and graph
            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                         log_device_placement=True))
            # self.saver = tf.train.Saver()

            # placeholder for inputs
            self.Ds_u = tf.placeholder(tf.float32, shape=[None, self.Ds_train.shape[1]])
            self.x_u = tf.placeholder(tf.float32, shape=[None, self.Rs_train.shape[1]])
            self.x_df = tf.placeholder(tf.float32, shape=[None, self.x_f.shape[1]])
            self.d_df = tf.placeholder(tf.float32, shape=[None, self.d_f.shape[1]])

             
            self.xequilibra_pred = self.net_u(self.Ds_u)


            self.h_pred = self.net_h(self.xequilibra_pred)

            self.x_res = self.net_f(self.d_df)

            self.aux = self.net_aux(self.x_df)

            self.constr =  self.net_constr(self.Ds_u)

            # loss
            # self.lossU0 = tf.reduce_mean(tf.square(self.x0_u - self.x0_pred))

            self.lossU = tf.reduce_mean(tf.square(self.x_u - self.xequilibra_pred))

            self.lossF = tf.reduce_mean(tf.square(self.x_res))

            self.lossaux = tf.norm(self.aux,ord=1,axis=0)

            self.lossconstr = tf.norm(self.constr,ord=1,axis=0)


            # self.loss = self.lossU + self.lossF + self.lossaux  #
            self.loss = self.lossU + 1*self.lossF + self.lossaux + self.lossconstr  #

            # Optimizer

            # self.optimizer = tf.train.AdamOptimizer(1e-3)
            # self.train_op_Adam = self.optimizer.minimize(self.loss)

            self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                    method='L-BFGS-B',
                                                                    options={'maxiter': 50000,
                                                                             'maxfun': 50000,
                                                                             'maxcor': 50,
                                                                             'maxls': 50,
                                                                             'ftol': 1.0 * np.finfo(float).eps})

            self.optimizer_Adam = tf.train.AdamOptimizer(0.001)
            self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

            init = tf.global_variables_initializer()
            self.sess.run(init)

        # Initialize the neural network
        def initialize_NN(self, layers):
            weights = []
            biases = []
            num_layers = len(layers)
            for l in range(0, num_layers - 1):
                W = self.xavier_init(size=[layers[l], layers[l + 1]])  # weights for the current layer
                b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32),
                                dtype=tf.float32)
                weights.append(W)
                biases.append(b)
            return weights, biases

        # generating weights
        def xavier_init(self, size):
            in_dim = size[0]
            out_dim = size[1]
            xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
            return  tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

                # tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev, dtype=tf.float32),
                #                dtype=tf.float32)

        # Architecture of the neural network
        def neural_net(self, Ds, weights, biases):
            num_layers = len(weights) + 1
            H = Ds
            # H = (Ds - self.lb) / (self.ub - self.lb)

            # H = 2.0 * (Ds - self.lb) / (self.ub - self.lb) - 1.0
            for l in range(0, num_layers - 2):
                W = weights[l]
                b = biases[l]
                H = tf.tanh(tf.add(tf.matmul(H, W), b))
            W = weights[-1]
            b = biases[-1]
            Y = tf.add(tf.matmul(H, W), b)
            return Y

        def net_u(self, Ds):
            output = self.neural_net(Ds, self.weights, self.biases)
            x = output ** 2
            # x = output[:, 0:1]
            return x

        def net_h(self, x):
            h = self.neural_net(x, self.weights_h, self.biases_h)
            bound_h = [tf.constant(0.25, dtype=tf.float32), tf.constant(0.45, dtype=tf.float32)]
            # return bound_h[0] + (bound_h[1] - bound_h[0]) * tf.sigmoid(h)
            # return tf.sigmoid(h)
            # return h
            return h**2

        def net_aux(self, x):

            NN = self.net_h(x)
            NN_x = tf.gradients(NN, x)[0]

            f_d = (-NN - x * NN_x) / ((1+x*NN)**2)

            f_aux = tf.maximum(tf.cast(0,dtype=tf.float32), f_d)
            return f_aux


        def net_constr(self, Ds):

            rho = self.rho
            lamda = self.lamda
            theta = self.theta

            x = self.net_u(Ds)

            p = tf.exp(-rho * Ds)
            h = self.net_h(x)
            Rx = 1 / (1 + x * h)
            xp = p * x
            hp = self.net_h(xp)
            Rp = 1 / (1 + xp * hp)
            F = lamda * p * x * (theta * Rx + (1 - theta) * Rp)

            F_x = tf.gradients(F, x)[0]
            f_d = tf.abs(F_x) - 1
            f_constr = tf.maximum(tf.cast(0,dtype=tf.float32), f_d)
            return f_constr


        """BH model"""
        def net_f(self, Ds):

            # load the parameters
            rho = self.rho
            lamda = self.lamda
            theta = self.theta

            #
            # betaI = self.net_BetaI(t, F)
            x = self.net_u(Ds)

            p = tf.exp(-rho * Ds)
            h = self.net_h(x)
            Rx = 1/(1+x*h)
            xp = p * x
            hp = self.net_h(xp)
            Rp = 1/(1+xp*hp)
            f_x = lamda * p * (theta * Rx + (1 - theta) * Rp) - 1

            return f_x

        def train(self, nIter):

            tf_dict = {self.Ds_u: self.Ds_train, self.d_df: self.d_f, self.x_df: self.x_f,
                       self.x_u: self.Rs_train
                       }  # ,self.d_df: self.d_f,

            start_time = time.time()
            for it in range(nIter + 1):
                self.sess.run(self.train_op_Adam, tf_dict)

                # Print
                if it % 100 == 0:
                    elapsed = time.time() - start_time
                    loss_value = self.sess.run(self.loss, tf_dict)
                    lossU_value = self.sess.run(self.lossU, tf_dict)
                    lossF_value = self.sess.run(self.lossF, tf_dict)
                    lossaux_value = self.sess.run(self.lossaux, tf_dict)
                    lossconstr_value = self.sess.run(self.lossconstr, tf_dict)
                    lamda_value = self.sess.run(self.lamda)
                    rho_value = self.sess.run(self.rho)
                    theta_value = self.sess.run(self.theta)

                    total_records.append(
                        np.array([it, loss_value, lossU_value, lossF_value]))
                    print(
                        'It: %d, Loss: %.3e, LossU: %.3e, LossF: %.3e, Lossaux: %.3e, Lossconstr: %.3e, rho: %.4e, Time: %.2f' %
                        (it, loss_value, lossU_value, lossF_value, lossaux_value, lossconstr_value, rho_value, elapsed))
                    start_time = time.time()

            # if LBFGS:
            # self.optimizer.minimize(self.sess,
            #                         feed_dict=tf_dict,
            #                         fetches=[self.loss, self.lossU0, self.lossU, self.lossF, self.beta, self.gamma],
            #                         loss_callback=self.callback)

        def predict(self, Ds_star):

            tf_dict = {self.Ds_u: Ds_star}
            # x_state = []
            x_state = self.sess.run(self.xequilibra_pred, tf_dict)
            h = self.sess.run(self.h_pred, tf_dict)
            # x_state = tf.reduce_mean(x[400:, :])

            # for i in range(Ds_star.shape[0]):
            #     x = self.sess.run(self.x_pred, tf_dict)
            #     x_state.append(x)
            # x_state.append(tf.reduce_mean(x[400:,:]))

            # x_stable = tf.convert_to_tensor(x_stable, dtype=tf.float32)

            return x_state, h


    if __name__ == "__main__":
        layers = [1] + 1 * [32] + [1]
        layers_h = [1] + 1 * [32] + [1]

        data_frame = pandas.read_excel('Data/Dataset2.xlsx')

        Dss = data_frame['Dss']
        Rs = data_frame['Rs']

        Ds_star = Dss
        Ds_star = Ds_star.to_numpy(dtype=np.float32)
        Ds_star = Ds_star[0:]

        Ds_star = Ds_star.reshape([len(Ds_star), 1])

        Rs_star = Rs
        # F_now_star = F_now_star.rolling(window=7).mean()
        Rs_star = Rs_star.to_numpy(dtype=np.float32)
        Rs_star = Rs_star[0:]

        Rs_star = Rs_star.reshape([len(Rs_star), 1])

        # lower and upper bounds
        lb = Ds_star.min(0)
        ub = Ds_star.max(0)

        sf = 1
        Rs_star = Rs_star * sf

        # Residual points
        N_f = 2000
        np.random.seed(42)
        d_f = lb + (ub - lb) * lhs(1, N_f)
        # d_f = np.linspace(lb, ub, num=N_f)

        x_f = np.linspace(lb, ub, num=N_f)
        # x_f = lb + (ub - lb) * lhs(1, N_f)

        N_pred = 100
        Ds_Pred = np.linspace(lb, ub, num=N_pred)

        ######################## Training and Predicting #####################
        ######################################################################
        Ds_train = Ds_star
        Rs_train = Rs_star

        from datetime import datetime

        now = datetime.now()
        dt_string = now.strftime("%y-%m-%d")

        current_directory = os.getcwd()
        experiment_directory = os.path.join(
            current_directory,
            'BHmodel',
            'DS2Bootstrap-results-' + dt_string
        )
        os.makedirs(experiment_directory, exist_ok=True)

        np.savetxt(
            os.path.join(experiment_directory, 'bootstrap_seeds.txt'),
            np.asarray(BOOTSTRAP_SEEDS, dtype=np.int64), fmt='%d'
        )

        #  baseline fit and centred residuals
        ######################################################################
        print('\n' + '=' * 70)
        print('Training baseline BH model')
        print('=' * 70)

        tf.reset_default_graph()
        set_random_seed(MODEL_INITIALIZATION_SEED)
        total_records = []

        baseline_model = PhysicsInformedNN(
            Ds_train, Rs_train, x_f, d_f, lb, ub,
            layers, layers_h, sf
        )
        baseline_model.train(10000)

        Rs_fitted, _ = baseline_model.predict(Ds_train)
        xstable_baseline, hx_baseline = baseline_model.predict(Ds_Pred)
        rho_baseline = baseline_model.sess.run(baseline_model.rho)
        lamda_baseline = baseline_model.sess.run(baseline_model.lamda)
        theta_baseline = baseline_model.sess.run(baseline_model.theta)

        residuals = Rs_train.reshape(-1) - Rs_fitted.reshape(-1)
        residuals = residuals - np.mean(residuals)

        pandas.DataFrame({
            'Ds_Pred': Ds_Pred.reshape(-1),
            'xstable': xstable_baseline.reshape(-1),
            'hx': hx_baseline.reshape(-1)
        }).to_csv(
            os.path.join(experiment_directory, 'baseline_predictions.csv'),
            index=False
        )

        pandas.DataFrame({
            'Ds': Ds_train.reshape(-1),
            'Rs_observed': Rs_train.reshape(-1),
            'Rs_fitted': Rs_fitted.reshape(-1),
            'centered_residual': residuals
        }).to_csv(
            os.path.join(experiment_directory, 'baseline_residuals.csv'),
            index=False
        )

        baseline_model.sess.close()
        del baseline_model

        # allocate result matrices
        ######################################################################
        xstable_bootstrap = np.full((N_pred, N_BOOTSTRAP), np.nan)
        hx_bootstrap = np.full((N_pred, N_BOOTSTRAP), np.nan)
        rho_bootstrap = np.full(N_BOOTSTRAP, np.nan)
        lamda_bootstrap = np.full(N_BOOTSTRAP, np.nan)
        theta_bootstrap = np.full(N_BOOTSTRAP, np.nan)
        bootstrap_summary = []

        #  residual Bootstrap and complete model refitting
        ######################################################################
        for bootstrap_index, bootstrap_seed in enumerate(BOOTSTRAP_SEEDS):
            run_number = bootstrap_index + 1
            print('\n' + '=' * 70)
            print('Bootstrap %d/%d, resampling seed = %d' %
                  (run_number, N_BOOTSTRAP, bootstrap_seed))
            print('=' * 70)

            run_directory = os.path.join(
                experiment_directory,
                'bootstrap_%03d_seed_%d' %
                (run_number, bootstrap_seed)
            )
            os.makedirs(run_directory, exist_ok=True)

            bootstrap_rng = np.random.RandomState(bootstrap_seed)
            sampled_indices = bootstrap_rng.randint(
                0, len(residuals), size=len(residuals)
            )
            sampled_residuals = residuals[sampled_indices]
            Rs_bootstrap = (
                Rs_fitted.reshape(-1) + sampled_residuals
            ).reshape(-1, 1).astype(np.float32)

            pandas.DataFrame({
                'Ds': Ds_train.reshape(-1),
                'Rs_bootstrap': Rs_bootstrap.reshape(-1),
                'sampled_residual_index': sampled_indices,
                'sampled_residual': sampled_residuals
            }).to_csv(
                os.path.join(run_directory,
                             'bootstrap_training_data.csv'),
                index=False
            )

            # Same initialization for each resample: the interval primarily
            # reflects observation/residual uncertainty, not optimizer noise.
            tf.reset_default_graph()
            set_random_seed(MODEL_INITIALIZATION_SEED)
            total_records = []
            model = None
            run_start = time.time()

            try:
                model = PhysicsInformedNN(
                    Ds_train, Rs_bootstrap, x_f, d_f, lb, ub,
                    layers, layers_h, sf
                )
                model.train(10000)

                rho_value = model.sess.run(model.rho)
                lamda_value = model.sess.run(model.lamda)
                theta_value = model.sess.run(model.theta)
                xstable_value, hx_value = model.predict(Ds_Pred)
                final_loss = model.sess.run(
                    model.loss,
                    {model.Ds_u: Ds_train,
                     model.x_u: Rs_bootstrap,
                     model.x_df: x_f,
                     model.d_df: d_f}
                )

                xstable_bootstrap[:, bootstrap_index] = \
                    xstable_value.reshape(-1)
                hx_bootstrap[:, bootstrap_index] = hx_value.reshape(-1)
                rho_bootstrap[bootstrap_index] = rho_value[0]
                lamda_bootstrap[bootstrap_index] = lamda_value[0]
                theta_bootstrap[bootstrap_index] = theta_value[0]

                pandas.DataFrame({
                    'Ds_Pred': Ds_Pred.reshape(-1),
                    'xstable': xstable_value.reshape(-1),
                    'hx': hx_value.reshape(-1)
                }).to_csv(
                    os.path.join(run_directory, 'prediction.csv'),
                    index=False
                )

                pandas.DataFrame(
                    np.asarray(total_records),
                    columns=['iteration', 'loss', 'lossU', 'lossF']
                ).to_csv(
                    os.path.join(run_directory, 'training_history.csv'),
                    index=False
                )

                bootstrap_summary.append({
                    'bootstrap': run_number,
                    'resampling_seed': bootstrap_seed,
                    'model_seed': MODEL_INITIALIZATION_SEED,
                    'status': 'success',
                    'rho': float(rho_value[0]),
                    'lamda': float(lamda_value[0]),
                    'theta': float(theta_value[0]),
                    'final_loss': float(final_loss),
                    'elapsed_seconds': time.time() - run_start
                })

            except Exception as error:
                print('Bootstrap %d failed: %s' %
                      (run_number, str(error)))
                with open(os.path.join(run_directory, 'error.txt'),
                          'w', encoding='utf-8') as error_file:
                    error_file.write(repr(error))
                bootstrap_summary.append({
                    'bootstrap': run_number,
                    'resampling_seed': bootstrap_seed,
                    'model_seed': MODEL_INITIALIZATION_SEED,
                    'status': 'failed',
                    'rho': np.nan,
                    'lamda': np.nan,
                    'theta': np.nan,
                    'final_loss': np.nan,
                    'elapsed_seconds': time.time() - run_start
                })

            finally:
                if model is not None:
                    model.sess.close()
                    del model

            # Update cumulative files after every run.
            pandas.DataFrame(bootstrap_summary).to_csv(
                os.path.join(experiment_directory,
                             'bootstrap_parameter_summary.csv'),
                index=False
            )

            completed_columns = [
                'bootstrap_%03d' % i
                for i in range(1, run_number + 1)
            ]
            xstable_frame = pandas.DataFrame(
                xstable_bootstrap[:, :run_number],
                columns=completed_columns
            )
            xstable_frame.insert(0, 'Ds_Pred', Ds_Pred.reshape(-1))
            xstable_frame.to_csv(
                os.path.join(experiment_directory,
                             'all_bootstrap_xstable.csv'),
                index=False
            )

            hx_frame = pandas.DataFrame(
                hx_bootstrap[:, :run_number],
                columns=completed_columns
            )
            hx_frame.insert(0, 'Ds_Pred', Ds_Pred.reshape(-1))
            hx_frame.to_csv(
                os.path.join(experiment_directory,
                             'all_bootstrap_hx.csv'),
                index=False
            )

        #  pointwise percentile-based 95% confidence intervals
        ######################################################################
        successful_runs = np.isfinite(rho_bootstrap)
        number_successful = int(np.sum(successful_runs))
        if number_successful < 2:
            raise RuntimeError(
                'Fewer than two Bootstrap runs completed successfully.'
            )

        xstable_lower = np.nanpercentile(
            xstable_bootstrap, 2.5, axis=1
        )
        # xstable_median = np.nanpercentile(
        #     xstable_bootstrap, 50.0, axis=1
        # )
        xstable_median = np.nanmean(
            xstable_bootstrap, axis=1
        )
        xstable_upper = np.nanpercentile(
            xstable_bootstrap, 97.5, axis=1
        )

        hx_lower_ds = np.nanpercentile(hx_bootstrap, 2.5, axis=1)
        # hx_median_ds = np.nanpercentile(hx_bootstrap, 50.0, axis=1)
        hx_median_ds = np.nanmean(hx_bootstrap, axis=1)
        hx_upper_ds = np.nanpercentile(hx_bootstrap, 97.5, axis=1)

        pandas.DataFrame({
            'Ds_Pred': Ds_Pred.reshape(-1),
            'baseline': xstable_baseline.reshape(-1),
            'bootstrap_median': xstable_median,
            'CI_2.5': xstable_lower,
            'CI_97.5': xstable_upper
        }).to_csv(
            os.path.join(experiment_directory,
                         'xstable_bootstrap_95CI.csv'),
            index=False
        )

        pandas.DataFrame({
            'Ds_Pred': Ds_Pred.reshape(-1),
            'baseline': hx_baseline.reshape(-1),
            'bootstrap_median': hx_median_ds,
            'CI_2.5': hx_lower_ds,
            'CI_97.5': hx_upper_ds
        }).to_csv(
            os.path.join(experiment_directory,
                         'hx_bootstrap_95CI_by_DsPred.csv'),
            index=False
        )

        pandas.DataFrame({
            'parameter': ['rho', 'lamda', 'theta'],
            'baseline': [float(rho_baseline[0]),
                         float(lamda_baseline[0]),
                         float(theta_baseline[0])],
            'bootstrap_mean': [np.nanmean(rho_bootstrap),
                               np.nanmean(lamda_bootstrap),
                               np.nanmean(theta_bootstrap)],
            'bootstrap_median': [np.nanmedian(rho_bootstrap),
                                 np.nanmedian(lamda_bootstrap),
                                 np.nanmedian(theta_bootstrap)],
            'CI_2.5': [np.nanpercentile(rho_bootstrap, 2.5),
                       np.nanpercentile(lamda_bootstrap, 2.5),
                       np.nanpercentile(theta_bootstrap, 2.5)],
            'CI_97.5': [np.nanpercentile(rho_bootstrap, 97.5),
                        np.nanpercentile(lamda_bootstrap, 97.5),
                        np.nanpercentile(theta_bootstrap, 97.5)]
        }).to_csv(
            os.path.join(experiment_directory,
                         'parameter_bootstrap_95CI.csv'),
            index=False
        )

        #  h(x) CI on a common xstable grid
        ######################################################################
        valid_xstable = xstable_bootstrap[:, successful_runs]
        valid_hx = hx_bootstrap[:, successful_runs]
        common_x_min = np.max(np.nanmin(valid_xstable, axis=0))
        common_x_max = np.min(np.nanmax(valid_xstable, axis=0))
        hx_ci_available = common_x_min < common_x_max

        if hx_ci_available:
            xstable_common = np.linspace(
                common_x_min, common_x_max, N_pred
            )
            hx_interpolated = np.full(
                (N_pred, number_successful), np.nan
            )

            for curve_index in range(number_successful):
                x_curve = valid_xstable[:, curve_index]
                h_curve = valid_hx[:, curve_index]
                finite = np.isfinite(x_curve) & np.isfinite(h_curve)
                x_curve = x_curve[finite]
                h_curve = h_curve[finite]
                order = np.argsort(x_curve)
                x_curve = x_curve[order]
                h_curve = h_curve[order]
                x_unique, unique_index = np.unique(
                    x_curve, return_index=True
                )
                h_unique = h_curve[unique_index]
                if len(x_unique) >= 2:
                    hx_interpolated[:, curve_index] = np.interp(
                        xstable_common, x_unique, h_unique
                    )

            hx_lower_x = np.nanpercentile(
                hx_interpolated, 2.5, axis=1
            )
            # hx_median_x = np.nanpercentile(
            #     hx_interpolated, 50.0, axis=1
            # )
            hx_median_x = np.nanmean(
                hx_interpolated, axis=1
            )
            hx_upper_x = np.nanpercentile(
                hx_interpolated, 97.5, axis=1
            )

            pandas.DataFrame({
                'xstable': xstable_common,
                'bootstrap_median': hx_median_x,
                'CI_2.5': hx_lower_x,
                'CI_97.5': hx_upper_x
            }).to_csv(
                os.path.join(experiment_directory,
                             'hx_bootstrap_95CI_by_xstable.csv'),
                index=False
            )

        
        #  uncertainty-band figures
        ######################################################################
        fig1, ax1 = plt.subplots(figsize=(7, 5))
        ax1.fill_between(
            Ds_Pred.reshape(-1), xstable_lower, xstable_upper,
            color='#78A7D8', alpha=0.35, linewidth=0,
            label='95% Bootstrap CI'
        )
        ax1.plot(
            Ds_Pred.reshape(-1), xstable_median,
            color='#1F5A9D', linewidth=2,
            label='Bootstrap median'
        )
        ax1.scatter(
            Ds_star.reshape(-1), Rs_star.reshape(-1),
            color='black', s=22, label='Observed data', zorder=3
        )
        ax1.set_xlabel(r'$D_s$')
        ax1.set_ylabel(r'$x_{stable}$')
        ax1.legend(frameon=False)
        fig1.tight_layout()
        fig1.savefig(
            os.path.join(experiment_directory,
                         'xstable_bootstrap_95CI.png'),
            dpi=600, bbox_inches='tight'
        )
        plt.close(fig1)

        if hx_ci_available:
            fig2, ax2 = plt.subplots(figsize=(7, 5))
            ax2.fill_between(
                xstable_common, hx_lower_x, hx_upper_x,
                color='#E89A78', alpha=0.35, linewidth=0,
                label='95% Bootstrap CI'
            )
            ax2.plot(
                xstable_common, hx_median_x,
                color='#B54120', linewidth=2,
                label='Bootstrap median'
            )
            ax2.set_xlabel(r'$x_{stable}$')
            ax2.set_ylabel(r'$h(x_{stable})$')
            ax2.legend(frameon=False)
            fig2.tight_layout()
            fig2.savefig(
                os.path.join(experiment_directory,
                             'hx_bootstrap_95CI.png'),
                dpi=600, bbox_inches='tight'
            )
            plt.close(fig2)

        print('\nBootstrap analysis completed.')
        print('Successful runs: %d/%d' %
              (number_successful, N_BOOTSTRAP))
        print('Results saved to:')
        print(experiment_directory)



