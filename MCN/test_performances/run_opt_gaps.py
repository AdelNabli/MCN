from MCN.test_performances.load_solved_instances import recover_instances, compute_opt_gap
from MCN.utils import load_saved_experts
import os

list_experts_small = load_saved_experts(os.path.join('models_cc', 'small', 'experts'))
list_experts_big = load_saved_experts(os.path.join('models_cc', 'big', 'experts'))
dict_instances = recover_instances('tables_MNC')
opt_gaps_big, val_exact, val_heur = compute_opt_gap(dict_instances, 3,3,3, list_experts_big)
opt_gaps_small, val_exact, val_heur = compute_opt_gap(dict_instances, 3,3,3, list_experts_big)

print(opt_gaps_big, opt_gaps_small)
