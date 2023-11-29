import os
import glob
import numpy as np
import json
from scipy.stats import bootstrap
import pdb

from collections import defaultdict
import trajectory.utils as utils

DATASETS = [
	f'{env}-{buffer}'
	for env in ['hopper', 'walker2d', 'halfcheetah', 'ant']
	for buffer in ['medium-expert-v2', 'medium-v2', 'medium-replay-v2']
]

LOGBASE = os.path.expanduser('~/logs')
TRIAL = '*'
EXP_NAME = 'plans/defaults/freq1_H1_beam50'

def load_results(paths, humanoid=False):
	'''
		paths : path to directory containing experiment trials
	'''
	scores = []
	infos = defaultdict(list)
	mean_infos = {}
	for i, path in enumerate(sorted(paths)):
		if humanoid:
			score, info = load_humanoid_result(path)
		else:
			score, info = load_result(path)
		if score is None:
			continue
		scores.append(score)
		for k, v in info.items():
			infos[k].append(v)

		suffix = path.split('/')[-1]

	for k, v in infos.items():
		mean_infos[k] = np.nanmean(v)

	res = bootstrap(data=[scores], statistic=np.mean, axis=0)
	bootstrap_error = res.standard_error
	conf_interval = res.confidence_interval
	mean_infos['bootstrap_error'] = bootstrap_error
	mean_infos['conf_lower'] = conf_interval.low
	mean_infos['conf_upper'] = conf_interval.high

	mean = np.mean(scores)
	err = np.std(scores) / np.sqrt(len(scores))
	return mean, err, scores, mean_infos


def load_humanoid_result(path):
	fullpath = os.path.join(path, 'rollout.json')
	suffix = path.split('/')[-1]

	if not os.path.exists(fullpath):
		return None, None

	results = json.load(open(fullpath, 'rb'))
	info = dict(returns=results["return"],
				discount_return=results["discount_return"],
				prediction_error=results["prediction_error"],
				value_mean=results["value_mean"],
				step=results["step"])

	return results["return"], info

def load_result(path):
	'''
		path : path to experiment directory; expects `rollout.json` to be in directory
	'''
	#path = os.path.join(path, "0")
	fullpath = os.path.join(path, 'rollout.json')
	suffix = path.split('/')[-1]

	if not os.path.exists(fullpath):
		return None, None

	results = json.load(open(fullpath, 'rb'))
	score = results['score']
	info = dict(returns=results["return"],
				first_value=results["first_value"],
				first_search_value=results["first_search_value"],
                discount_return=results["discount_return"],
				prediction_error=results["prediction_error"],
				step=results["step"])

	return score * 100, info

#######################
######## setup ########
#######################

class Parser(utils.Parser):
	dataset: str = None
	exp_name: str = None
	output: str = None
	test_planner: str = None
	wildcard_exp_name: bool = True

if __name__ == '__main__':

	args = Parser().parse_args()

	write_to_file = args.output is not None

	if args.wildcard_exp_name:
		exp_name = args.exp_name+"*"
	else:
		exp_name = args.exp_name

	if write_to_file:
		f = open(args.output, "a")

	for dataset in ([args.dataset] if args.dataset else DATASETS):
		subdirs = glob.glob(os.path.join(LOGBASE, dataset))

		for subdir in subdirs:
			reldir = subdir.split('/')[-1]
			if args.test_planner is not None:
				paths = glob.glob(os.path.join(subdir, exp_name, TRIAL, args.test_planner))
			else:
				paths = glob.glob(os.path.join(subdir, exp_name, TRIAL))

			if "mocapact" in args.dataset:
				mean, err, returns, infos = load_results(paths, humanoid=True)
				string_print=f'{args.exp_name} | {dataset.ljust(30)} | {len(returns)} returns | return {mean:.2f} +/- {err:.2f} | value mean {infos["value_mean"]:.2f}'
			else:
				mean, err, scores, infos = load_results(paths)
				string_print=f'{dataset.ljust(30)} | {len(scores)} scores | score {mean:.2f} +/- {err:.2f} | '
			for k, v in infos.items():
				string_print += f'{k} {v:.4f} | '
			print(string_print)
			if write_to_file:
				f.write(string_print+'\n')

	if write_to_file:
		f.close()
			
