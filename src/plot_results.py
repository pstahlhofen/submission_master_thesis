import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import json
import os
import numpy as np

def compare_results(result_path, max_area_bisection):
	if result_path[-1]!='/':
		result_path += '/'
	genetic_path = result_path + 'Basic_GA/'
	has_genetic_results	= os.path.isdir(genetic_path)
	combined_path = result_path + 'Enhanced_GA/'
	has_combined_results = os.path.isdir(combined_path)
	if not has_genetic_results and not has_combined_results:
		raise RuntimeError(
			f'Cannot compare results because {result_path}'
			f'does neither contain a subdirectory "Basic_GA"'
			f'nore "Enhanced_GA".'
		)
	fig = plt.figure()
	# All y-values are multiplied by 10000 to convert m^2 to cm^2
	plt.axhline(max_area_bisection*10000, label='bisection')
	if has_genetic_results:
		best_areas_genetic = get_best_areas(genetic_path)
		x = list(range(1, len(best_areas_genetic)+1))
		plt.scatter(x, best_areas_genetic*10000, label='basic genetic')
	if has_combined_results:
		best_areas_combined = get_best_areas(combined_path)
		x = list(range(1, len(best_areas_combined)+1))
		plt.scatter(x, best_areas_combined*10000, label='enhanced genetic')
	# Make sure the x-axis uses only integer values
	ax = fig.gca()
	ax.xaxis.set_major_locator(MaxNLocator(integer=True))
	plt.xlabel('Trial')
	plt.ylabel('Maximal Leak Area in cm^2')
	plt.legend()
	plt.savefig(result_path + 'result_comparison.png', dpi=300)

def get_best_areas(path):
		subfolders = [f.name for f in os.scandir(path) if f.is_dir()]
		trials = [f for f in subfolders if f.split('-')[0]=='Trial']
		trials = sorted(trials, key=lambda f: int(f.split('-')[-1]))
		trials = [path + trial + '/' for trial in trials]
		best_areas = []
		for trial in trials:
			best_solution_file = trial + 'best_solution.json'
			with open(best_solution_file, 'r') as fp:
				best_solution_dict = json.load(fp)
			best_areas.append(best_solution_dict['area'])
		return np.array(best_areas)

def print_means(small_dataset_path, large_dataset_path):
	areas_genetic_small = get_best_areas(small_dataset_path + 'Basic_GA/')
	areas_combined_small = get_best_areas(small_dataset_path + 'Enhanced_GA/')
	areas_genetic_large = get_best_areas(large_dataset_path + 'Basic_GA/')
	areas_combined_large = get_best_areas(large_dataset_path + 'Enhanced_GA/')
	print(f'Mean Genetic Small: {areas_genetic_small.mean()}')
	print(f'Mean Combined Small: {areas_combined_small.mean()}')
	print(f'Mean Genetic Large: {areas_genetic_large.mean()}')
	print(f'Mean Combined Large: {areas_combined_large.mean()}')

if __name__=='__main__':
#	compare_results('../Results/Hanoi_1week/', 0.0125)
	small_dataset_path = '../Results/Hanoi_1week/'
	large_dataset_path = '../Results/Hanoi_2week/'
	print_means(small_dataset_path, large_dataset_path)
