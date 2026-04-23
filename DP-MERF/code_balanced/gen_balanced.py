import json
import os
import torch as pt
from torch.optim.lr_scheduler import StepLR
import argparse
import numpy as np
from models_gen import FCCondGen, ConvCondGen
from aux_funcs import plot_mnist_batch, log_args, flatten_features, log_final_score
from data_loading import get_dataloaders
from rff_mmd_approx import get_rff_losses
from synth_data_benchmark import test_gen_data, test_passed_gen_data, datasets_colletion_def
from synth_data_2d import plot_data
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math
import scipy

import sys

sys.path.append("..")
from utils.histogram_voting import get_info_gaussian, vote, get_embeddings


def train_single_release(gen, device, optimizer, epoch, rff_mmd_loss, log_interval, batch_size, n_data):
	n_iter = n_data // batch_size
	for batch_idx in range(n_iter):
		gen_code, gen_labels = gen.get_code(batch_size, device)
		loss = rff_mmd_loss(gen(gen_code), gen_labels)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if batch_idx % log_interval == 0:
			print("Train Epoch: {} [{}/{}]\tLoss: {:.6f}".format(epoch, batch_idx * batch_size, n_data, loss.item()))


def compute_rff_loss(gen, data, labels, rff_mmd_loss, device):
	bs = labels.shape[0]
	gen_code, gen_labels = gen.get_code(bs, device)
	gen_samples = gen(gen_code)
	return rff_mmd_loss(data, labels, gen_samples, gen_labels)


def train_multi_release(gen, device, train_loader, optimizer, epoch, rff_mmd_loss, log_interval):

	for batch_idx, (data, labels) in enumerate(train_loader):
		data, labels = data.to(device), labels.to(device)
		data = flatten_features(data)

		loss = compute_rff_loss(gen, data, labels, rff_mmd_loss, device)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if batch_idx % log_interval == 0:
			n_data = len(train_loader.dataset)
			print("Train Epoch: {} [{}/{}]\tLoss: {:.6f}".format(epoch, batch_idx * len(data), n_data, loss.item()))


def log_gen_data(gen, device, epoch, n_labels, log_dir):
	ordered_labels = pt.repeat_interleave(pt.arange(n_labels), n_labels)[:, None].to(device)
	gen_code, _ = gen.get_code(100, device, labels=ordered_labels)
	gen_samples = gen(gen_code).detach()

	plot_samples = gen_samples[:100, ...].cpu().numpy()
	plot_mnist_batch(plot_samples, 10, n_labels, log_dir + f"samples_ep{epoch}", denorm=False)


def get_losses(ar, train_loader, device, n_feat, n_labels):
	# if ar.loss_type == 'real_mmd':
	#   minibatch_loss = get_real_mmd_loss(ar.rff_sigma, n_labels, ar.batch_size)
	#   single_release_loss = None
	# elif ar.loss_type == 'kmeans':
	#   minibatch_loss = None
	#   single_release_loss = get_kmeans_mmd_loss(train_loader, n_labels, ar.tgt_epsilon, ar.n_means,
	#                                             ar.rff_sigma, ar.batch_size, ar.dp_kmeans_encoding_dim)
	#
	# elif ar.loss_type == 'rff':
	single_release_loss, minibatch_loss, _ = get_rff_losses(train_loader, n_feat, ar.d_rff, ar.rff_sigma, device, n_labels, ar.noise_factor, ar.mmd_type)
	# else:
	#   raise ValueError

	return minibatch_loss, single_release_loss


def get_args():
	parser = argparse.ArgumentParser()

	# BASICS
	parser.add_argument("--seed", type=int, default=None, help="sets random seed")
	parser.add_argument("--log-interval", type=int, default=100, help="print updates after n steps")
	parser.add_argument("--base-log-dir", type=str, default="logs/gen/", help="path where logs for all runs are stored")
	parser.add_argument("--log-name", type=str, default=None, help="subdirectory for this run")
	parser.add_argument("--log-dir", type=str, default=None, help="override save path. constructed if None")
	parser.add_argument("--data", type=str, default="digits", help="options are digits, fashion and 2d")
	parser.add_argument("--create-dataset", action="store_true", default=True, help="if true, make 60k synthetic code_balanced")

	# OPTIMIZATION
	parser.add_argument("--batch-size", "-bs", type=int, default=500)
	parser.add_argument("--test-batch-size", "-tbs", type=int, default=1000)
	parser.add_argument("--gen-batch-size", "-gbs", type=int, default=1000)
	parser.add_argument("--epochs", "-ep", type=int, default=5)
	parser.add_argument("--lr", "-lr", type=float, default=0.01, help="learning rate")
	parser.add_argument("--lr-decay", type=float, default=0.9, help="per epoch learning rate decay factor")

	# MODEL DEFINITION
	# parser.add_argument('--batch-norm', action='store_true', default=True, help='use batch norm in model')
	parser.add_argument("--conv-gen", action="store_true", default=True, help="use convolutional generator")
	parser.add_argument("--d-code", "-dcode", type=int, default=5, help="random code dimensionality")
	parser.add_argument("--gen-spec", type=str, default="200", help="specifies hidden layers of generator")
	parser.add_argument("--kernel-sizes", "-ks", type=str, default="5,5", help="specifies conv gen kernel sizes")
	parser.add_argument("--n-channels", "-nc", type=str, default="16,8", help="specifies conv gen kernel sizes")

	# DP SPEC
	parser.add_argument("--d-rff", type=int, default=10_000, help="number of random filters for apprixmate mmd")
	parser.add_argument("--rff-sigma", "-rffsig", type=str, default=None, help="standard dev. for filter sampling")
	parser.add_argument("--noise-factor", "-noise", type=float, default=5.0, help="privacy noise parameter")

	# ALTERNATE MODES
	parser.add_argument("--single-release", action="store_true", default=True, help="get 1 data mean embedding only")

	parser.add_argument("--loss-type", type=str, default="rff", help="how to approx mmd", choices=["rff", "kmeans", "real_mmd"])
	# parser.add_argument('--real-mmd', action='store_true', default=False, help='for debug: dont approximate mmd')
	# parser.add_argument('--kmeans-mmd', action='store_true', default=False, help='for debug: dont approximate mmd')

	parser.add_argument("--n-means", type=int, default=10, help="number of means to find per class")
	parser.add_argument("--dp-kmeans-encoding-dim", type=int, default=10, help="dimension the data is projected to")
	parser.add_argument("--tgt-epsilon", type=float, default=1.0, help="privacy epsilon for dp k-means")
	parser.add_argument("--kmeans-delta", type=float, default=0.01, help="soft failure probability in dp k-means")
	parser.add_argument("--mmd-type", type=str, default="sphere", help="how to approx mmd", choices=["sphere", "r+r"])

	parser.add_argument("--center-data", action="store_true", default=False, help="k-means requires centering")

	# synth_d2 data
	parser.add_argument("--synth-spec-string", type=str, default="disc_k5_n10000_row5_col5_noise0.2", help="")
	parser.add_argument("--test-split", type=float, default=0.1, help="only relevant for synth_2d so far")

	# ============== VOTING ARGUMENTS ==============
	parser.add_argument("--skip_vote", action="store_true", default=False)
	parser.add_argument("--vote_rounds", type=int, default=1)
	parser.add_argument("--n_splits", type=int, default=1, help="number of independent voting runs to union")
	parser.add_argument("--num_synth_factor", type=float, default=1, help="proportion of synthetic data to generate, relative to original")
	parser.add_argument("--epsilon_vote", type=float, default=1.0)
	parser.add_argument("--model_path", type=str, default="pt_models/epsilon_1.0/gen.pt")

	# Pool composition: generator_fraction + random_fraction must equal 1.0
	# These control what proportion of each batch of candidates comes from
	# the DP-MERF generator vs uniform random sampling.
	parser.add_argument("--generator_fraction", type=float, default=1.0,
						help="fraction of candidate pool from DP-MERF generator. "
							 "Must sum to 1 with random_fraction.")
	parser.add_argument("--random_fraction", type=float, default=0.0,
						help="fraction of candidate pool from uniform random sampling. "
							 "Must sum to 1 with generator_fraction.")

	# oversample_factor: how many candidates to generate relative to how many
	# we need to keep for this split.
	# e.g. n_per_class_split=100, oversample_factor=1.25 -> generate 125 candidates,
	# vote down to 100. Must be >= 1.0.
	# Recommended: oversample_factor <= n_splits, so that candidates per split
	# do not exceed the number of private points per class.
	parser.add_argument("--oversample_factor", type=float, default=1.25,
						help="ratio of candidates generated to samples needed per split. "
							 "Must be >= 1.0. Higher = more filtering but worse voting quality "
							 "if candidates exceed private points per class.")

	ar = parser.parse_args()

	preprocess_args(ar)
	log_args(ar.log_dir, ar)
	return ar


def find_required_noise_multiplier(epsilon, num_steps, num_N):
    delta= 1/(num_N*math.log(num_N))
    def delta_Gaussian(eps, mu):
        """Compute delta of Gaussian mechanism with shift mu or equivalently noise scale 1/mu"""
        if mu==0:
            return 0
        return scipy.stats.norm.cdf(-eps / mu + mu / 2) - np.exp(eps) * scipy.stats.norm.cdf(-eps / mu - mu / 2)
    def eps_Gaussian(delta, mu):
        """Compute eps of Gaussian mechanism with shift mu or equivalently noise scale 1/mu"""
        def f(x):
            return delta_Gaussian(x, mu) - delta
        return scipy.optimize.root_scalar(f, bracket=[0, 500], method='brentq').root
    def compute_epsilon(noise_multiplier, num_steps, delta):
        return eps_Gaussian(delta, np.sqrt(num_steps) / noise_multiplier)
    def objective(x):
        return -compute_epsilon(x[0], num_steps, delta)
    def constraints(x):
        return (epsilon - .00001) - compute_epsilon(x[0], num_steps, delta)

    output = scipy.optimize.minimize(lambda x: objective(x), x0=[1], bounds=[(0, None)], constraints={'type': 'ineq', 'fun': constraints})
    assert(output.success)
    assert(-output.fun <= epsilon + 1e-4)
    return output.x[0]


def preprocess_args(ar):
	ar.base_log_dir = os.path.abspath(ar.base_log_dir) + "/"

	# validate pool composition fractions
	assert abs(ar.generator_fraction + ar.random_fraction - 1.0) < 1e-6, \
		f"generator_fraction ({ar.generator_fraction}) + random_fraction ({ar.random_fraction}) must equal 1.0"
	assert ar.oversample_factor >= 1.0, \
		f"oversample_factor must be >= 1.0, got {ar.oversample_factor}"

	if ar.log_dir is None:
		if ar.log_name is None:
			ar.log_name = (
				f"dpmerf_{ar.data}"
				f"_ep{ar.epochs}"
				f"_noise{ar.noise_factor}"
			)
		base = ar.base_log_dir + ar.log_name + "/"
		if ar.skip_vote:
			ar.log_dir = base + "no_vote/"
		else:
			ar.log_dir = os.path.join(
				base,
				f"eps_{ar.epsilon_vote}",
				f"syn_{ar.num_synth_factor}",
				f"splits_{ar.n_splits}",
				f"rounds_{ar.vote_rounds}",
				f"over_{ar.oversample_factor}",
				f"genfrac_{ar.generator_fraction}",
			) + "/"
		ar.model_pt_path = base + "gen.pt"
	else:
		ar.model_pt_path = ar.log_dir + "gen.pt"

	if ar.model_path is not None:
		ar.model_pt_path = ar.model_path

	os.makedirs(ar.log_dir, exist_ok=True)

	if ar.seed is None:
		ar.seed = np.random.randint(0, 1000)
	assert ar.data in {"digits", "fashion", "2d"}
	if ar.rff_sigma is None:
		ar.rff_sigma = "105" if ar.data == "digits" else "127"

	if ar.loss_type == "kmeans" and ar.tgt_epsilon > 0.0:
		assert ar.center_data, "dp kmeans requires centering of data"

	if ar.data == "2d":
		ar.conv_gen = False
	else:
		ar.conv_gen = True


def run_voting_pipeline(gen, device, ar, n_per_class, private_embeddings_per_class,
						columns, info, n_labels, gen_batch_size, noise_multiplier_vote,
						log_dir=None, split_idx=0):
	"""
	Runs the full voting pipeline for one split.
	Returns syn_per_class dict {label: [samples]} with n_per_class per label.

	Each vote round:
	  1. Generate n_candidates = n_per_class * oversample_factor candidates per class.
	     Mix of generator and random per generator_fraction / random_fraction.
	     From round 2 onwards, survivors from the previous round are added too.
	  2. Vote: private points vote for nearest candidate, keep top n_per_class.
	"""
	n_candidates  = int(n_per_class * ar.oversample_factor)
	n_from_gen    = int(n_candidates * ar.generator_fraction)
	n_from_rand   = n_candidates - n_from_gen  # ensures exact total

	# start with empty pool — first round generates everything fresh
	syn_per_class = {label: [] for label in range(n_labels)}

	for i in range(ar.vote_rounds):
		for label in range(n_labels):
			candidates = []

			if n_from_gen > 0:
				gen_samples = synthesize_data_for_label(
					gen, device, label, n_from_gen, gen_batch_size=gen_batch_size
				)
				candidates.extend(gen_samples.tolist())

			if n_from_rand > 0:
				rand_samples = np.random.uniform(
					[info["x"]["min"], info["y"]["min"]],
					[info["x"]["max"], info["y"]["max"]],
					size=(n_from_rand, 2)
				).tolist()
				candidates.extend(rand_samples)

			# from round 2 onwards, survivors from previous round join the pool
			if i > 0:
				candidates.extend(syn_per_class[label])

			public_embeddings = get_embeddings(candidates, columns, info)
			best, _, _ = vote(
				public=candidates,
				public_embeddings=public_embeddings,
				private_embeddings=private_embeddings_per_class[label],
				count=n_per_class,
				noise_multiplier=noise_multiplier_vote
			)
			syn_per_class[label] = best

		# plot intermediate results if log_dir provided
		if log_dir is not None:
			syn_data_iter   = np.array([s for label in range(n_labels) for s in syn_per_class[label]])
			syn_labels_iter = np.array([label for label in range(n_labels) for _ in syn_per_class[label]])
			data_tuple_iter = datasets_colletion_def(
				syn_data_iter, syn_labels_iter, None, None, None, None
			)
			iter_dir = os.path.join(log_dir, f"split_{split_idx}_iteration_{i}")
			os.makedirs(iter_dir, exist_ok=True)
			plot_curr(data_tuple_iter, iter_dir, f"split_{split_idx}_iteration_{i}")

	return syn_per_class


def synthesize_data_with_uniform_labels(gen, device, gen_batch_size=1000, n_data=60000, n_labels=10):
	gen.eval()
	if n_data % gen_batch_size != 0:
		assert n_data % 100 == 0
		gen_batch_size = n_data // 100
	assert gen_batch_size % n_labels == 0
	n_iterations = n_data // gen_batch_size

	data_list = []
	ordered_labels = pt.repeat_interleave(pt.arange(n_labels), gen_batch_size // n_labels)[:, None].to(device)
	labels_list = [ordered_labels] * n_iterations

	with pt.no_grad():
		for idx in range(n_iterations):
			gen_code, gen_labels = gen.get_code(gen_batch_size, device, labels=ordered_labels)
			gen_samples = gen(gen_code)
			data_list.append(gen_samples)
	return pt.cat(data_list, dim=0).cpu().numpy(), pt.cat(labels_list, dim=0).cpu().numpy()


def synthesize_data_for_label(gen, device, label, n_samples, gen_batch_size=1000):
	gen.eval()
	data_list = []
	remaining = n_samples
	with pt.no_grad():
		while remaining > 0:
			batch = min(gen_batch_size, remaining)
			label_tensor = pt.full((batch, 1), label, dtype=pt.long).to(device)
			gen_code, gen_labels = gen.get_code(batch, device, labels=label_tensor)
			gen_samples = gen(gen_code)
			data_list.append(gen_samples)
			remaining -= batch
	return pt.cat(data_list, dim=0).cpu().numpy()


def test_results(data_key, log_name, log_dir, data_tuple, eval_func):
	if data_key in {"digits", "fashion"}:
		final_score = test_gen_data(log_name, data_key, subsample=0.1, custom_keys="logistic_reg", data_from_torch=True)
		log_final_score(log_dir, final_score)
	elif data_key == "2d":
		final_score = test_passed_gen_data(log_name, data_tuple, log_save_dir=None, log_results=False, subsample=0.1, custom_keys="mlp", compute_real_to_real=True)
		log_final_score(log_dir, final_score)
		eval_score = eval_func(data_tuple.x_gen, data_tuple.y_gen.flatten())
		print(f"Score of evaluation function: {eval_score}")
		with open(os.path.join(log_dir, "eval_score"), "w") as f:
			f.writelines([f"{eval_score}"])

		plot_data(data_tuple.x_real_train, data_tuple.y_real_train.flatten(), os.path.join(log_dir, "plot_train"), center_frame=True)
		plot_data(data_tuple.x_gen, data_tuple.y_gen.flatten(), os.path.join(log_dir, "plot_gen"))
		plot_data(data_tuple.x_gen, data_tuple.y_gen.flatten(), os.path.join(log_dir, "plot_gen_sub0.2"), subsample=0.2)
		plot_data(data_tuple.x_gen, data_tuple.y_gen.flatten(), os.path.join(log_dir, "plot_gen_centered"), center_frame=True)


def plot_curr(data_tuple, log_dir, title):
	# plot_data(data_tuple.x_gen, data_tuple.y_gen.flatten(), os.path.join(log_dir, "plot_gen"))
	plot_data(data_tuple.x_gen, data_tuple.y_gen.flatten(), os.path.join(log_dir, "plot_gen_sub0.2"), subsample=0.2, title=title)
	plot_data(data_tuple.x_gen, data_tuple.y_gen.flatten(), os.path.join(log_dir, "plot_gen_centered"), center_frame=True, title=title)


def main():
	# load settings
	ar = get_args()
	pt.manual_seed(ar.seed)
	np.random.seed(ar.seed)
	use_cuda = pt.cuda.is_available()
	device = pt.device("cuda" if use_cuda else "cpu")

	# load data
	data_pkg = get_dataloaders(ar.data, ar.batch_size, ar.test_batch_size, use_cuda, ar.center_data, ar.synth_spec_string, ar.test_split)
	# init model
	if ar.conv_gen:
		gen = ConvCondGen(ar.d_code, ar.gen_spec, data_pkg.n_labels, ar.n_channels, ar.kernel_sizes).to(device)
	else:
		use_sigmoid = ar.data in {"digits", "fashion"}
		gen = FCCondGen(ar.d_code, ar.gen_spec, data_pkg.n_features, data_pkg.n_labels, use_sigmoid=use_sigmoid, batch_norm=True).to(device)

	if os.path.isfile(ar.model_pt_path):
		print("Existing model found, loading...")
		gen.load_state_dict(pt.load(ar.model_pt_path, map_location=device))
	else:
		print("No existing model found, training from scratch...")
		minibatch_loss, single_release_loss = get_losses(ar, data_pkg.train_loader, device, data_pkg.n_features, data_pkg.n_labels)
		# init optimizer
		optimizer = pt.optim.Adam(list(gen.parameters()), lr=ar.lr)
		scheduler = StepLR(optimizer, step_size=1, gamma=ar.lr_decay)

		# training loop
		for epoch in range(1, ar.epochs + 1):
			if ar.single_release:
				train_single_release(gen, device, optimizer, epoch, single_release_loss, ar.log_interval, ar.batch_size, data_pkg.n_data)
			else:
				train_multi_release(gen, device, data_pkg.train_loader, optimizer, epoch, minibatch_loss, ar.log_interval)

			# testing doesn't really inform how training is going, so it's commented out
			# test(gen, device, test_loader, rff_mmd_loss, epoch, ar.batch_size, ar.log_dir)
			if ar.data in {"digits", "fashion"}:
				log_gen_data(gen, device, epoch, data_pkg.n_labels, ar.log_dir)
			scheduler.step()

		# save trained model and data
		pt.save(gen.state_dict(), ar.model_pt_path)

	if ar.create_dataset:
		data_id = "synthetic_mnist" if ar.data in {"digits", "fashion"} else "gen_data"

		if ar.skip_vote:
			print("DP-MERF Default mode, no voting")
			syn_data, syn_labels = synthesize_data_with_uniform_labels(gen, device, gen_batch_size=ar.gen_batch_size, n_data=data_pkg.n_data, n_labels=data_pkg.n_labels)
		else:
			print("DP-MERF with DP-Histogram voting")
			num_samples_to_generate = int(data_pkg.n_data * ar.num_synth_factor)
			n_per_class       = int(num_samples_to_generate / data_pkg.n_labels)
			n_per_class_split = n_per_class // ar.n_splits

			# warn if voting quality may be poor
			n_private_per_class    = data_pkg.n_data // data_pkg.n_labels
			n_candidates_per_split = int(n_per_class_split * ar.oversample_factor)
			if n_candidates_per_split > n_private_per_class:
				print(f"WARNING: candidates per split ({n_candidates_per_split}) exceeds "
					  f"private points per class ({n_private_per_class}). "
					  f"Consider reducing oversample_factor or increasing n_splits.")

			print(f"  n_per_class:        {n_per_class}")
			print(f"  n_splits:           {ar.n_splits}")
			print(f"  n_per_class_split:  {n_per_class_split}")
			print(f"  oversample_factor:  {ar.oversample_factor}")
			print(f"  candidates/split:   {n_candidates_per_split}")
			print(f"  private/class:      {n_private_per_class}")

			columns = {"numerical": ["x", "y"], "categorical": [], "label": "label"}

			real_data = data_pkg.train_data.data
			real_labels = data_pkg.train_data.targets
			if isinstance(real_data, pt.Tensor):
				real_data = real_data.cpu().numpy()
				real_labels = real_labels.cpu().numpy().flatten().astype(int)

			info = get_info_gaussian(real_data, columns["numerical"])

			private_embeddings_per_class = {}
			for label in range(data_pkg.n_labels):
				real_class = real_data[real_labels == label].tolist()
				private_embeddings_per_class[label] = get_embeddings(real_class, columns, info)

			# num_steps = vote_rounds * n_splits: total times private data is accessed.
			# splits are sequential (same private data), classes are parallel (disjoint subsets = free).
			voting_noise = find_required_noise_multiplier(
				ar.epsilon_vote,
				num_steps=ar.vote_rounds * ar.n_splits,
				num_N=data_pkg.n_data
			)
			print(f"VOTING NOISE: {voting_noise}")

			all_syn_per_class = {label: [] for label in range(data_pkg.n_labels)}
			for split_idx in range(ar.n_splits):
				print(f"  [Split {split_idx + 1}/{ar.n_splits}]")
				split_result = run_voting_pipeline(
					gen, device, ar,
					n_per_class=n_per_class_split,
					private_embeddings_per_class=private_embeddings_per_class,
					columns=columns,
					info=info,
					n_labels=data_pkg.n_labels,
					gen_batch_size=ar.gen_batch_size,
					noise_multiplier_vote=voting_noise,
					log_dir=ar.log_dir,
					split_idx=split_idx
				)
				for label in range(data_pkg.n_labels):
					all_syn_per_class[label].extend(split_result[label])

			syn_data   = np.array([s for label in range(data_pkg.n_labels) for s in all_syn_per_class[label]])
			syn_labels = np.array([label for label in range(data_pkg.n_labels) for _ in all_syn_per_class[label]])

		np.savez(ar.log_dir + data_id, data=syn_data, labels=syn_labels)

		data_tuple = datasets_colletion_def(syn_data, syn_labels,
											data_pkg.train_data.data, data_pkg.train_data.targets,
											data_pkg.test_data.data, data_pkg.test_data.targets)
		test_results(ar.data, ar.log_name, ar.log_dir, data_tuple, data_pkg.eval_func)
		# final_score = test_gen_data(ar.log_name, ar.data, subsample=0.1, custom_keys='logistic_reg')
		# log_final_score(ar.log_dir, final_score)


if __name__ == "__main__":
	main()
