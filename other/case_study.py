import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE
import torch.nn.functional as F
import os

# =============================
#  一、配置与加载模型
# =============================

# MODEL_PATH = '/mnt/e/LRM/code/SIGMA-main/model_file/SASRec_ml-1m.pth'
# MODEL_PATH = '/mnt/e/LRM/code/SIGMA-main/model_file/EchoMamba4Rec_ml-1m.pth'
# MODEL_PATH = '/mnt/e/LRM/code/SIGMA-main/model_file/Mamba4Rec_ml-1m.pth'
# MODEL_PATH = '/mnt/e/LRM/code/SIGMA-main/model_file/BERT4Rec_ml-1m.pth'
# MODEL_PATH = '/mnt/e/LRM/code/SIGMA-main/model_file/GRU4Rec_ml-1m.pth'
MODEL_PATH = '/mnt/e/LRM/code/SIGMA-main/model_file/DiSAM_ml-1m.pth'
# MODEL_PATH = '/mnt/e/LRM/code/SIGMA-main/model_file/SIGMA_ml-1m.pth'
MODEL_NAME = MODEL_PATH.split("/")[-1].split("_")[0]
print(MODEL_NAME)
SAVE_DIR = f'./case_study_figs/{MODEL_PATH.split("/")[-1].split(".")[0]}'
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from gated_mamba import SIGMA  # 修改为你模型的路径
from recbole.model.sequential_recommender import GRU4Rec
from recbole.model.sequential_recommender import BERT4Rec
from recbole.model.sequential_recommender import SASRec
from mamba4rec import Mamba4Rec
from EchoMamba4Rec import EchoMamba4Rec
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
import recbole.model.sequential_recommender as seq_models


def load_model(config_path):
    seq_models.SIGMA = SIGMA
    cfg = Config(model=SIGMA, config_file_list=[config_path])
    dataset = create_dataset(cfg)
    train_data, valid_data, test_data = data_preparation(cfg, dataset)
    model = SIGMA(cfg, dataset).to(DEVICE)

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    print("✅ Model loaded successfully.")
    return model, dataset, cfg


# =============================
#  二、预测与比对函数
# =============================

def predict_for_user(model, dataset, user_id, topk=10, num_targets=10):
    """
    对指定用户进行预测，可指定多个真实目标项目。
    Args:
        user_id: 用户ID
        topk: 预测Top-K项目
        num_targets: 取最后num_targets个item作为真实目标（默认1个）
    """
    # 获取用户的交互历史

    user_mask = (dataset.inter_feat['user_id'] == user_id)
    user_inter = dataset.inter_feat[user_mask]

    if len(user_inter) < 3:
        print(f"⚠️ 用户 {user_id} 交互太少（仅 {len(user_inter)} 条），无法进行预测。")
        return None, None

    # 提取张量并按时间排序
    item_ids = user_inter['item_id'].numpy()
    if 'timestamp' in user_inter:
        timestamps = user_inter['timestamp'].numpy()
        sorted_idx = np.argsort(timestamps)
        item_ids = item_ids[sorted_idx]

    # 保证 num_targets 不超过序列长度
    num_targets = min(num_targets, len(item_ids) - 1)

    # 使用最后 num_targets 个 item 作为真实目标
    target_items = item_ids[-num_targets:]
    item_seq = torch.tensor(item_ids[:-num_targets]).unsqueeze(0).to(DEVICE)
    item_seq_len = torch.tensor([item_seq.size(1)]).to(DEVICE)

    # 构建 interaction
    interaction = {
        model.USER_ID: torch.tensor([user_id]).to(DEVICE),
        model.ITEM_SEQ: item_seq,
        model.ITEM_SEQ_LEN: item_seq_len
    }
    device = next(model.parameters()).device
    print("Model on:", device)

    for k, v in interaction.items():
        print(f"{k}: {v.device}")
    # 预测所有item的得分
    with torch.no_grad():
        scores = model.full_sort_predict(interaction)  # (1, n_items)
        topk_items = torch.topk(scores, k=topk).indices.squeeze(0).cpu().numpy()

    return target_items, topk_items


def compare_prediction_with_truth(target_item, topk_items):
    """输出预测与真实结果的比对"""
    print("\n==================== Prediction Result ====================")
    print(f"🎯 True next item: {target_item}")
    print(f"🔮 Top-{len(topk_items)} predicted items: {topk_items.tolist()}")
    if target_item in topk_items:
        rank = np.where(topk_items == target_item)[0][0] + 1
        print(f"✅ Hit! True item ranked at position {rank}.")
    else:
        print("❌ Missed. True item not in Top-K.")


# =============================
#  三、可视化函数区
# =============================

def visualize_attention_heatmap(attention_scores, user_id, title="Attention Heatmap"):
    plt.figure(figsize=(6, 5))
    sns.heatmap(attention_scores, cmap='YlGnBu')
    plt.title(f'{title} for User {user_id}')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}/{title.replace(" ", "_").lower()}_user{user_id}.png', dpi=300)
    plt.close()
    print(f"🧠 {title} saved for user {user_id}.")


def visualize_item_embedding(model, MODEL_NAME):
    item_emb = model.item_embedding.weight.detach().cpu().numpy()
    emb_2d = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(item_emb)
    plt.figure(figsize=(6, 5))
    plt.scatter(emb_2d[:, 0], emb_2d[:, 1], s=4, alpha=0.6)
    plt.title(f"t-SNE of Item Embeddings ({MODEL_NAME})")
    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}/tsne_items.png', dpi=300)
    plt.close()
    print("🎨 Item embedding t-SNE saved.")


# =============================
#  四、主函数
# =============================

def main():
    config_path = 'config.yaml'
    model, dataset, cfg = load_model(config_path)

    # --- Step 1: 用户预测与比对 ---
    user_id = 5050
    target_item, topk_items = predict_for_user(model, dataset, user_id, topk=10)
    if target_item is not None:
        compare_prediction_with_truth(target_item, topk_items)

    # --- Step 2: 继续可视化 ---
    visualize_item_embedding(model, MODEL_NAME)



if __name__ == "__main__":
    main()
