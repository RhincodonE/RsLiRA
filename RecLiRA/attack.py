import os
import numpy as np
import pandas as pd
import argparse
from scipy.stats import norm
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_auc_score


# Suppose these are your arrays:


def fit_normal_distribution(data_array):
    mean = np.mean(data_array)
    variance = np.var(data_array)
    return mean, variance

def load_user_level_log_odds(directory, model_name, seed):
    train_file = f'{directory}/in_{model_name}.csv'
    out_file = f'{directory}/out_{model_name}.csv'
    train_df = pd.read_csv(train_file)
    out_df = pd.read_csv(out_file)

    train_users = train_df['user_id'].tolist()
    out_users = out_df['user_id'].tolist()
    train_prob_log_odds = train_df['prob_log_odds'].values
    out_prob_log_odds = out_df['prob_log_odds'].values

    print(f'Loaded user-level train log odds from {train_file}')
    print(f'Loaded user-level out log odds from {out_file}')

    return train_prob_log_odds, out_prob_log_odds, train_users, out_users

def load_item_level_log_odds(directory, model_name, seed):
    train_file = f'{directory}/in_{model_name}.csv'
    out_file = f'{directory}/out_{model_name}.csv'

    train_df = pd.read_csv(train_file)
    out_df = pd.read_csv(out_file)

    train_users = train_df['user_id'].tolist()
    out_users = out_df['user_id'].tolist()
    train_prob_log_odds = train_df['prob_log_odds'].values
    out_prob_log_odds = out_df['prob_log_odds'].values

    print(f'Loaded item-level train log odds from {train_file}')
    print(f'Loaded item-level out log odds from {out_file}')

    return train_prob_log_odds, out_prob_log_odds, train_users, out_users

def load_distribution(filename):
    """
    Load in_mean, in_variance, out_mean, out_variance, best_threshold from a CSV file.
    """
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        in_mean = df['in_mean'].values[0]
        in_variance = df['in_variance'].values[0]
        out_mean = df['out_mean'].values[0]
        out_variance = df['out_variance'].values[0]
        # If your CSV has threshold column, uncomment the following line:
        # threshold = df['best_threshold'].values[0]
        # Otherwise, set it to None:
        threshold = None

        print(f"Loaded distribution data from {filename}")
        return in_mean, in_variance, out_mean, out_variance, threshold
    else:
        print(f"File {filename} does not exist.")
        return None, None, None, None, None

def find_optimal_threshold_from_cdfs(
    train_users,
    out_users,
    train_prob_log_odds,
    out_prob_log_odds,
    out_mean,
    out_variance,
    threshold_loaded=None,
    output_path='./results/performance.png'
):
    """
    Computes TPR/FPR using roc_curve on continuous "cdf scores" without building a huge NxN matrix.
    Then:
      1. Plots a log-log ROC curve and saves it.
      2. Writes (FPR, TPR) to a .csv file.
      3. Returns the best threshold by a chosen metric (e.g., max TPR-FPR) and the AUC.
    """

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc

    # Compute the OUT distribution CDF
    out_std = np.sqrt(out_variance)
    from scipy.stats import norm
    def out_cdf(x):
        return norm.cdf(x, loc=out_mean, scale=out_std)

    # 1) Compute continuous scores for in/out
    train_cdfs = out_cdf(train_prob_log_odds)
    out_cdfs   = out_cdf(out_prob_log_odds)

    # 2) Build label and score arrays
    scores = np.concatenate([train_cdfs, out_cdfs])
    labels = np.concatenate([np.ones(len(train_cdfs)), np.zeros(len(out_cdfs))])

    # 3) Use scikit-learn’s roc_curve to get the full ROC
    fpr, tpr, thresholds = roc_curve(labels, scores)

    # 4) Compute AUC (area under the curve)
    roc_auc = auc(fpr, tpr)

    # 5) Find the “best threshold” if desired
    #    For example, using Youden's J statistic = TPR - FPR:
    best_thr = None
    best_j   = float('-inf')
    for i in range(len(thresholds)):
        j = tpr[i] - fpr[i]
        if j > best_j:
            best_j  = j
            best_thr = thresholds[i]

    # If you had a pre-loaded threshold, you can still use it
    # or compare it to best_thr, but it’s generally optional now.
    # threshold_loaded would override if that’s your pipeline logic.

    # 6) Plot and save the ROC curve (log-log)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f"ROC (AUC = {roc_auc:.4f})")

    # Log scales
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim([1e-5, 1.0])
    plt.ylim([1e-5, 1.0])
    plt.xlabel('False Positive Rate (log scale)')
    plt.ylabel('True Positive Rate (log scale)')
    plt.title('ROC Curve (log-log scale)')
    plt.legend(loc="lower right")
    plt.grid(True)

    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"ROC curve saved to {output_path}")

    # 7) Save (FPR, TPR) to CSV
    csv_filename = output_path.replace('.png', '.csv')
    df_roc = pd.DataFrame({'FPR': fpr, 'TPR': tpr})
    df_roc.to_csv(csv_filename, index=False)
    print(f"(FPR, TPR) saved to {csv_filename}")

    # Return the best threshold found, plus the AUC
    return best_thr, roc_auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='1m_Neu_MF_seed0')
    parser.add_argument('--seed', type=int, default=24)
    args = parser.parse_args()

    # 1. User-level data
    train_prob_log_odds_user, out_prob_log_odds_user, train_users_user, out_users_user = load_user_level_log_odds(
        './user-data/target', args.model_name, args.seed
    )
    in_mean_user, in_variance_user = fit_normal_distribution(train_prob_log_odds_user)
    out_mean_user, out_variance_user = fit_normal_distribution(out_prob_log_odds_user)

    # Load user-level distribution from distribution_user_shadow.csv
    in_mean_loaded_user, in_variance_loaded_user, out_mean_loaded_user, out_variance_loaded_user, threshold_user_loaded = load_distribution(
        './distribution/distribution_user_shadow.csv'
    )

    # 2. Item-level data
    train_prob_log_odds_item, out_prob_log_odds_item, train_users_item, out_users_item = load_item_level_log_odds(
        './item-data/target', args.model_name, args.seed
    )
    in_mean_item, in_variance_item = fit_normal_distribution(train_prob_log_odds_item)
    out_mean_item, out_variance_item = fit_normal_distribution(out_prob_log_odds_item)

    # Load item-level distribution from distribution_item_shadow.csv
    in_mean_loaded_item, in_variance_loaded_item, out_mean_loaded_item, out_variance_loaded_item, threshold_item_loaded = load_distribution(
        './distribution/distribution_item_shadow.csv'
    )

    arrs = [train_prob_log_odds_user, out_prob_log_odds_user,
            train_prob_log_odds_item, out_prob_log_odds_item]

    for i, arr in enumerate(arrs, 1):
        if np.any(np.isnan(arr)):
            print(f"Array #{i} contains NaN at indices:", np.where(np.isnan(arr)))
        if np.any(np.isinf(arr)):
            print(f"Array #{i} contains INF at indices:", np.where(np.isinf(arr)))

    # User-level computations
    _, auc_user = find_optimal_threshold_from_cdfs(
        train_users_user,
        out_users_user,
        train_prob_log_odds_user,
        out_prob_log_odds_user,
        out_mean_loaded_user,
        out_variance_loaded_user,
        threshold_loaded=threshold_user_loaded,
        output_path='./results/user-level-performance.png'
    )
    print(f"User-level AUC Score: {auc_user}")

    # Item-level computations
    _, auc_item = find_optimal_threshold_from_cdfs(
        train_users_item,
        out_users_item,
        train_prob_log_odds_item,
        out_prob_log_odds_item,
        out_mean_loaded_item,
        out_variance_loaded_item,
        threshold_loaded=threshold_item_loaded,
        output_path='./results/item-level-performance.png'
    )
    print(f"Item-level AUC Score: {auc_item}")

if __name__ == "__main__":
    main()
