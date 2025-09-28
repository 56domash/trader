# # """
# # check_features.py
# # 元の features.py と新しい features_packs.py の出力を比較するスクリプト
# # """

# # import sqlite3
# # import pandas as pd

# # from tq.features import compute_packs as compute_packs_old
# # from tq.features_packs import compute_packs as compute_packs_new

# # DB_PATH = "./runtime.db"
# # SYMBOL = "7203.T"
# # DATE = "2025-09-10"

# # def load_bars(db_path, symbol, date):
# #     """DBからbars_1mをロードして指定日のデータを返す"""
# #     with sqlite3.connect(db_path) as conn:
# #         q = f"""
# #         SELECT ts, open, high, low, close, volume
# #         FROM bars_1m
# #         WHERE symbol = ?
# #           AND ts BETWEEN ? AND ?
# #         ORDER BY ts
# #         """
# #         start = date + "T00:00:00"
# #         end   = date + "T23:59:59"
# #         df = pd.read_sql(q, conn, params=(symbol, start, end), parse_dates=["ts"])
# #         df.set_index("ts", inplace=True)
# #     return df

# # def main():
# #     # --- データ準備
# #     df = load_bars(DB_PATH, SYMBOL, DATE)

# #     # --- 両方の特徴量を計算
# #     df_old = compute_packs_old(df.copy())
# #     df_new = compute_packs_new(df.copy())

# #     # --- 列の比較
# #     cols_old = set(df_old.columns)
# #     cols_new = set(df_new.columns)

# #     print("=== 列の差分 ===")
# #     print("旧のみ:", cols_old - cols_new)
# #     print("新のみ:", cols_new - cols_old)

# #     # --- 共通列の値比較
# #     print("\n=== 値の比較 (共通列) ===")
# #     diffs = []
# #     for col in sorted(cols_old & cols_new):
# #         diff = (df_old[col] - df_new[col]).abs().max()
# #         diffs.append((col, diff))
# #         if diff > 1e-9:
# #             print(f"[差分あり] {col}: max diff = {diff}")
# #         else:
# #             print(f"[OK] {col}")

# #     # --- サマリー
# #     diff_cols = [c for c,d in diffs if d > 1e-9]
# #     print("\n=== サマリー ===")
# #     print(f"共通列: {len(cols_old & cols_new)} 列")
# #     print(f"差分あり: {len(diff_cols)} 列 → {diff_cols}")

# # if __name__ == "__main__":
# #     main()
# import pandas as pd
# # 変更後（名前衝突を避けて別名に）
# from tq.features import compute_packs as compute_old
# from tq.features_packs import compute_packs as compute_new


# def main():
#     # データロード
#     df = pd.read_parquet("sample.parquet")

#     # 例: 旧=features.py / 新=features_packs.py を比較
#     df_old = compute_old(df.copy(), use_pack1=True,
#                          use_pack2=True, use_pack3=True, use_pack4=True)
#     df_new = compute_new(df.copy())

#     # --- 🔹 インデックスの重複を除去 ---
#     df_old = df_old.loc[~df_old.index.duplicated(keep="first")].copy()
#     df_new = df_new.loc[~df_new.index.duplicated(keep="first")].copy()

#     # --- 🔹 インデックスを揃える ---
#     df_old, df_new = df_old.align(df_new, join="inner", axis=0)

#     # --- 列の差分チェック ---
#     old_cols = set(df_old.columns)
#     new_cols = set(df_new.columns)

#     print("=== 列の差分 ===")
#     print("旧のみ:", old_cols - new_cols)
#     print("新のみ:", new_cols - old_cols)
#     print()

#     # --- 共通列の比較 ---
#     common_cols = old_cols & new_cols
#     print("=== 値の比較 (共通列) ===")
#     diffs = []
#     for col in sorted(common_cols):
#         try:
#             diff = (df_old[col] - df_new[col]).abs().max()
#             if diff > 1e-6:
#                 print(f"[差分あり] {col}: max diff = {diff}")
#                 diffs.append(col)
#             else:
#                 print(f"[OK] {col}")
#         except Exception as e:
#             print(f"[ERROR] {col}: {e}")

#     print()
#     print("=== サマリー ===")
#     print(f"共通列: {len(common_cols)} 列")
#     print(f"差分あり: {len(diffs)} 列 → {diffs}")


# if __name__ == "__main__":
#     main()
import pandas as pd
# ← features.py 側の compute_packs を呼ぶ
from tq.features import compute_packs as compute_old
from tq.features_packs import compute_packs as compute_new


def main():
    df = pd.read_parquet("sample.parquet")

    # 旧: features.py
    df_old = compute_old(df.copy(), use_pack1=True,
                         use_pack2=True, use_pack3=True, use_pack4=True)

    # 新: features_packs.py
    df_new = compute_new(df.copy())

    # --- 重複indexを除去 ---
    df_old = df_old.loc[~df_old.index.duplicated(keep="first")].copy()
    df_new = df_new.loc[~df_new.index.duplicated(keep="first")].copy()

    # --- インデックスを揃える ---
    df_old, df_new = df_old.align(df_new, join="inner", axis=0)

    # --- 列の差分チェック ---
    old_cols = set(df_old.columns)
    new_cols = set(df_new.columns)
    print("=== 列の差分 ===")
    print("旧のみ:", old_cols - new_cols)
    print("新のみ:", new_cols - old_cols)
    print()

    # --- 共通列の比較 ---
    common_cols = old_cols & new_cols
    print("=== 値の比較 (共通列) ===")
    diffs = []
    for col in sorted(common_cols):
        try:
            diff = (df_old[col] - df_new[col]).abs().max()
            if diff > 1e-6:
                print(f"[差分あり] {col}: max diff = {diff}")
                diffs.append(col)
            else:
                print(f"[OK] {col}")
        except Exception as e:
            print(f"[ERROR] {col}: {e}")

    print()
    print("=== サマリー ===")
    print(f"共通列: {len(common_cols)} 列")
    print(f"差分あり: {len(diffs)} 列 → {diffs}")


if __name__ == "__main__":
    main()
