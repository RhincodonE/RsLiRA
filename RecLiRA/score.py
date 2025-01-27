import os
import glob
import pandas as pd
from collections import defaultdict
import argparse
import numpy as np
from scipy.stats import norm
from utils import compute_user_item_specific_thresholds,compute_user_specific_thresholds,load_user_distribution,load_item_distribution,load_all_pairs,load_all_item_pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='1m_Neu_MF_seed18')
    parser.add_argument('--shadow_model_name', type=str, default='ml-1m_Neu_MF_1')
    parser.add_argument('--model_path', type=str, default='./target_models')
    parser.add_argument('--distribution_path', type=str, default='./distribution')
    parser.add_argument('--removal', type=int, default=0)
    parser.add_argument('--user_id', type=int, default=1698)
    parser.add_argument('--removal_top_k', type=float, default=0.1)
    args = parser.parse_args()

    # Load user-level data and distribution
    user_data = load_all_pairs('./user-data/target/')
    in_mean_user, in_variance_user, out_mean_user, out_variance_user = load_user_distribution(args)
    if in_mean_user is not None:
        user_metrics = compute_user_specific_thresholds(out_mean_user, out_variance_user, user_data)
        user_results = []
        for uid, (threshold, TPR_user, FPR_user) in user_metrics.items():
            user_results.append((uid, TPR_user, FPR_user))

        os.makedirs('./results', exist_ok=True)
        user_df = pd.DataFrame(user_results, columns=["user_id", "TPR", "FPR"])
        if args.removal==1:
            user_df.to_csv(f'./results/scores-user-{args.user_id}-{args.removal_top_k}.csv', index=False)
        else:
            user_df.to_csv('./results/scores-user-base.csv', index=False)
        print("User-level results saved to ./results/scores-user.csv")

    # Load item-level data and distribution
    user_item_data = load_all_item_pairs('./item-data/target/')
    in_mean_item, in_variance_item, out_mean_item, out_variance_item = load_item_distribution(args)
    if in_mean_item is not None:
        item_metrics = compute_user_item_specific_thresholds(out_mean_item, out_variance_item, user_item_data)
        item_df = pd.DataFrame(item_metrics, columns=["user_id", "item_id", "TPR", "FPR"])
        if args.removal==1:
            item_df.to_csv(f'./results/scores-item-{args.user_id}-{args.removal_top_k}.csv', index=False)
        else:
            item_df.to_csv('./results/scores-item-base.csv', index=False)
        print("Item-level results saved to ./results/scores_item.csv")

if __name__ == "__main__":
    main()
