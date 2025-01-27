import os
import random
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
import argparse
from tqdm import tqdm

# Keep only what is used
from utils import (
    download_data,
    load_data,
    amazon_load,
    hit,
    ndcg,
    collect_probability_differences_by_user,
    calculate_log_odds,
    mkdir,
    calculate_log_odds_sum
)
from dataset import NCFDataHandler
from models import NeuMF


def train_and_evaluate_model(args, model, train_loader, test_loader, device, writer):
    """
    Train and evaluate the NeuMF model.
    Saves the best model based on HR@top_k.
    """
    best_hr = 0
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_function = nn.BCELoss()

    for epoch in range(1, args.epochs + 1):
        model.train()  # Enable dropout (if applicable)
        start_time = time.time()
        train_loss = 0.0

        # Training loop with progress bar
        print(f"Epoch {epoch}/{args.epochs} - Training:")
        train_progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch}", leave=False)

        for user, item, label in train_progress_bar:
            user = user.to(device)
            item = item.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            prediction = model(user, item)
            loss = loss_function(prediction, label)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            # Update the progress bar with the current loss
            train_progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss /= len(train_loader)
        writer.add_scalar('loss/Train_loss', train_loss, epoch)

        # Evaluation loop with progress bar
        print(f"Epoch {epoch}/{args.epochs} - Evaluation:")
        model.eval()
        HR, NDCG = [], []
        eval_loss = 0.0

        eval_progress_bar = tqdm(test_loader, desc=f"Evaluating Epoch {epoch}", leave=False)
        for user, item, label in eval_progress_bar:
            user = user.to(device)
            item = item.to(device)

            predictions = model(user, item)
            loss = loss_function(predictions, label.to(device))
            eval_loss += loss.item()

            # Top-K items
            _, indices = torch.topk(predictions, args.top_k)
            recommends = torch.take(item, indices).cpu().numpy().tolist()

            # Leave-one-out: there's one positive item per user
            ng_item = item[0].item()
            HR.append(hit(ng_item, recommends))
            NDCG.append(ndcg(ng_item, recommends))

            eval_progress_bar.set_postfix(eval_loss=f"{loss.item():.4f}")

        eval_loss /= len(test_loader)
        HR_mean = np.mean(HR)
        NDCG_mean = np.mean(NDCG)
        writer.add_scalar('Perfomance/HR@10', HR_mean, epoch)
        writer.add_scalar('Perfomance/NDCG@10', NDCG_mean, epoch)
        writer.add_scalar('loss/Eval_loss', eval_loss, epoch)

        elapsed_time = time.time() - start_time
        print(
            f"Epoch {epoch} | Time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))} "
            f"| Train Loss: {train_loss:.4f} | Eval Loss: {eval_loss:.4f} "
            f"| HR@{args.top_k}: {HR_mean:.4f} | NDCG@{args.top_k}: {NDCG_mean:.4f}"
        )

        # Save the best model
        if HR_mean > best_hr:
            best_hr = HR_mean
            if args.out:
                if not os.path.exists(args.model_path):
                    os.mkdir(args.model_path)
                torch.save(model.state_dict(), f"{args.model_path}/{args.model_name}.pth")

    writer.close()
    return best_hr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_ng', type=int, default=4)
    parser.add_argument('--num_ng_test', type=int, default=99)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--factor_num', type=int, default=32)
    parser.add_argument('--layers', type=list, default=[64, 32, 16, 8])
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--model_name', type=str, default='ml-1m_Neu_MF')
    parser.add_argument('--model_path', type=str, default='./target_models')
    parser.add_argument('--out', action='store_true', default=True)
    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--dataset', type=str, default='ml-1m')
    args = parser.parse_args()

    # Set device
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # Tensorboard writer
    writer = SummaryWriter()

    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Download/load dataset
    MAIN_PATH = './content/'
    if args.dataset == 'ml-1m':
        DATA_URL = "https://raw.githubusercontent.com/sparsh-ai/rec-data-public/master/ml-1m-dat/ratings.dat"
        DATA_PATH = os.path.join(MAIN_PATH, 'ratings.dat')
        download_data(DATA_URL, DATA_PATH)
        dataset = load_data(DATA_PATH)
    elif args.dataset == 'Amazon-DigitalMusic':
        DATA_PATH = os.path.join(MAIN_PATH, 'ratings_Digital_Music.csv')
        dataset = amazon_load(DATA_PATH)
    elif args.dataset == 'Amazon-Beauty':
        DATA_PATH = os.path.join(MAIN_PATH, 'ratings_Beauty.csv')
        dataset = amazon_load(DATA_PATH)
    else:
        raise ValueError("Unsupported dataset specified.")

    # Number of users/items
    num_users = dataset['user_id'].nunique() + 1
    num_items = dataset['item_id'].nunique() + 1

    # Data handlers and loaders
    data_handler = NCFDataHandler(args, dataset)
    train_loader = data_handler.get_train_instance()
    test_loader = data_handler.get_test_instance()
    # These methods must exist in your NCFDataHandler
    in_loader = data_handler.get_in_instance()
    out_loader = data_handler.get_out_instance()

    # Initialize and train model
    model = NeuMF(args, num_users, num_items).to(device)
    best_hr = train_and_evaluate_model(args, model, train_loader, test_loader, device, writer)
    print(f"Best HR: {best_hr}")

    # ---------------------------
    # Collect user/item log-odds
    # ---------------------------
    print("\nCollecting probability differences (in/out sets)...")
    in_prob_diff, in_user_item_diff = collect_probability_differences_by_user(model, in_loader, device, flag="sum")
    out_prob_diff, out_user_item_diff = collect_probability_differences_by_user(model, out_loader, device, flag="sum")

    # User-level log-odds
    in_prob_diff_values = np.array(list(in_prob_diff.values()))
    out_prob_diff_values = np.array(list(out_prob_diff.values()))
    in_prob_log_odds = calculate_log_odds_sum(in_prob_diff_values)
    out_prob_log_odds = calculate_log_odds_sum(out_prob_diff_values)

    # Make directories
    mkdir('./user-data')
    mkdir('./item-data')

    # Save user-level data
    if args.model_path == './target_models':
        os.makedirs('user-data/target', exist_ok=True)
        in_log_file = f'user-data/target/in_{args.model_name}.csv'
        out_log_file = f'user-data/target/out_{args.model_name}.csv'

        in_users = list(in_prob_diff.keys())
        out_users = list(out_prob_diff.keys())

        in_df = pd.DataFrame({'user_id': in_users, 'prob_log_odds': in_prob_log_odds})
        out_df = pd.DataFrame({'user_id': out_users, 'prob_log_odds': out_prob_log_odds})

        in_df.to_csv(in_log_file, index=False)
        out_df.to_csv(out_log_file, index=False)

        print(f'Saved train log odds to {in_log_file}')
        print(f'Saved out log odds to {out_log_file}')

        # Save item-level data to item-data/target
        os.makedirs('item-data/target', exist_ok=True)

        # Flatten in/out item probabilities
        in_item_rows = []
        for uid, item_pairs in in_user_item_diff.items():
            for it, prob in item_pairs:
                in_item_rows.append((uid, it, prob))

        out_item_rows = []
        for uid, item_pairs in out_user_item_diff.items():
            for it, prob in item_pairs:
                out_item_rows.append((uid, it, prob))

        # Compute item-level log-odds
        in_item_probs = np.array([row[2] for row in in_item_rows])
        out_item_probs = np.array([row[2] for row in out_item_rows])
        in_item_log_odds = calculate_log_odds(in_item_probs)
        out_item_log_odds = calculate_log_odds(out_item_probs)

        # Create DataFrames
        in_item_df = pd.DataFrame({
            'user_id': [r[0] for r in in_item_rows],
            'item_id': [r[1] for r in in_item_rows],
            'prob_log_odds': in_item_log_odds
        })
        out_item_df = pd.DataFrame({
            'user_id': [r[0] for r in out_item_rows],
            'item_id': [r[1] for r in out_item_rows],
            'prob_log_odds': out_item_log_odds
        })

        # Save item-level CSV
        in_item_file = f'item-data/target/in_{args.model_name}.csv'
        out_item_file = f'item-data/target/out_{args.model_name}.csv'
        in_item_df.to_csv(in_item_file, index=False)
        out_item_df.to_csv(out_item_file, index=False)

        print(f'Saved item-level in data to {in_item_file}')
        print(f'Saved item-level out data to {out_item_file}')

    else:
        # If the model_path differs, treat as shadow
        mkdir('user-data/shadow')
        in_log_file = f'user-data/shadow/in_{args.model_name}.csv'
        out_log_file = f'user-data/shadow/out_{args.model_name}.csv'
        np.savetxt(in_log_file, in_prob_log_odds, delimiter=',')
        np.savetxt(out_log_file, out_prob_log_odds, delimiter=',')
        print(f'Saved train log odds to {in_log_file}')
        print(f'Saved out log odds to {out_log_file}')

        # Save item-level data to "shadow"
        mkdir('item-data/shadow')
        in_item_rows = []
        for uid, item_pairs in in_user_item_diff.items():
            for it, prob in item_pairs:
                in_item_rows.append((uid, it, prob))

        out_item_rows = []
        for uid, item_pairs in out_user_item_diff.items():
            for it, prob in item_pairs:
                out_item_rows.append((uid, it, prob))

        in_item_probs = np.array([row[2] for row in in_item_rows])
        out_item_probs = np.array([row[2] for row in out_item_rows])
        in_item_log_odds = calculate_log_odds(in_item_probs)
        out_item_log_odds = calculate_log_odds(out_item_probs)

        in_item_df = pd.DataFrame({
            'user_id': [r[0] for r in in_item_rows],
            'item_id': [r[1] for r in in_item_rows],
            'prob_log_odds': in_item_log_odds
        })
        out_item_df = pd.DataFrame({
            'user_id': [r[0] for r in out_item_rows],
            'item_id': [r[1] for r in out_item_rows],
            'prob_log_odds': out_item_log_odds
        })

        in_item_file = f'item-data/shadow/in_{args.model_name}.csv'
        out_item_file = f'item-data/shadow/out_{args.model_name}.csv'
        in_item_df.to_csv(in_item_file, index=False)
        out_item_df.to_csv(out_item_file, index=False)

        print(f'Saved item-level in data to {in_item_file}')
        print(f'Saved item-level out data to {out_item_file}')


if __name__ == "__main__":
    main()
