import re
import io
import csv
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import random

import pandas as pd
import streamlit as st

JST = timezone(timedelta(hours=9))

# 例：
# Received Notification: <Notification from username:楓さん。, sender user id:usr_... to ... of type: requestInvite, ... created at: 12/21/2025 15:40:13 UTC, ... type:requestInvite, ...>
PATTERN = re.compile(
    r"Received Notification:\s*<Notification\s+from\s+username:(?P<username>.*?),\s*"
    r"sender user id:(?P<sender_user_id>usr_[0-9a-fA-F\-]+).*?"
    r"created at:\s*(?P<created_at>\d{1,2}/\d{1,2}/\d{4}\s+\d{2}:\d{2}:\d{2})\s+UTC,.*?"
    r"type:requestInvite",
    re.DOTALL
)

def parse_created_at_utc_to_jst(s: str) -> datetime:
    # "12/21/2025 15:40:13" (UTC) -> JST datetime aware
    dt_utc = datetime.strptime(s, "%m/%d/%Y %H:%M:%S").replace(tzinfo=timezone.utc)
    return dt_utc.astimezone(JST)

def parse_log_to_df(text: str) -> pd.DataFrame:
    rows = []
    for m in PATTERN.finditer(text):
        username = (m.group("username") or "").strip()
        sender_user_id = (m.group("sender_user_id") or "").strip()
        created_at_raw = (m.group("created_at") or "").strip()
        created_at_jst = parse_created_at_utc_to_jst(created_at_raw)
        rows.append(
            {
                "created_at_jst": created_at_jst,
                "username": username,
                "sender_user_id": sender_user_id,
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("created_at_jst").reset_index(drop=True)
    return df

def normalize_blacklist(df_bl: pd.DataFrame) -> pd.DataFrame:
    # columns: sender_user_id / username のどちらでもOK
    cols = {c.strip(): c for c in df_bl.columns}
    sender_col = cols.get("sender_user_id")
    user_col = cols.get("username")
    if sender_col is None and user_col is None:
        # 1列だけのCSVなども許容（最初の列をID扱い）
        if len(df_bl.columns) >= 1:
            sender_col = df_bl.columns[0]

    out = pd.DataFrame()
    if sender_col is not None:
        out["sender_user_id"] = df_bl[sender_col].astype(str).str.strip()
    else:
        out["sender_user_id"] = ""

    if user_col is not None:
        out["username"] = df_bl[user_col].astype(str).str.strip()
    else:
        out["username"] = ""

    out = out.fillna("")
    out = out[(out["sender_user_id"] != "") | (out["username"] != "")]
    out = out.drop_duplicates().reset_index(drop=True)
    return out

def apply_time_filter(df: pd.DataFrame, start_jst: datetime, end_jst: datetime) -> pd.DataFrame:
    if df.empty:
        return df
    # Streamlitのdatetime_inputがnaiveを返す場合があるのでJSTを付与
    if start_jst.tzinfo is None:
        start_jst = start_jst.replace(tzinfo=JST)
    if end_jst.tzinfo is None:
        end_jst = end_jst.replace(tzinfo=JST)

    mask = (df["created_at_jst"] >= start_jst) & (df["created_at_jst"] <= end_jst)
    return df.loc[mask].reset_index(drop=True)

def apply_blacklist(df: pd.DataFrame, bl: pd.DataFrame) -> pd.DataFrame:
    if df.empty or bl.empty:
        return df

    # sender_user_id一致を優先で除外（usernameだけのBLも一応対応）
    bl_ids = set(bl["sender_user_id"].astype(str).str.strip())
    bl_names = set(bl["username"].astype(str).str.strip())

    def is_blocked(r):
        sid = str(r["sender_user_id"]).strip()
        un = str(r["username"]).strip()
        return (sid in bl_ids and sid != "") or (un in bl_names and un != "")

    mask = ~df.apply(is_blocked, axis=1)
    return df.loc[mask].reset_index(drop=True)

def dedup_participants(df: pd.DataFrame) -> pd.DataFrame:
    # 同一ユーザーが複数回出てきても「参加者」は1人扱いにする（必要なら後で切替）
    if df.empty:
        return df
    return df.drop_duplicates(subset=["sender_user_id"], keep="first").reset_index(drop=True)

def draw_winners(df_participants: pd.DataFrame, k: int, seed: int | None) -> pd.DataFrame:
    if df_participants.empty or k <= 0:
        return df_participants.iloc[0:0].copy()

    k = min(k, len(df_participants))
    rng = random.Random(seed)  # Python標準random = メルセンヌ・ツイスタ
    idx = list(df_participants.index)
    winners_idx = rng.sample(idx, k=k)
    winners = df_participants.loc[winners_idx].copy()
    winners = winners.sort_values("created_at_jst").reset_index(drop=True)
    return winners

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")

st.set_page_config(page_title="抽選機プロト", layout="wide")
st.title("抽選機（Streamlitプロト）")

with st.sidebar:
    st.header("入力")
    log_file = st.file_uploader("logファイル（.txt / .log）", type=["txt", "log"])
    num_winners = st.number_input("抽選人数", min_value=1, max_value=1000, value=3, step=1)

    st.caption("時間はJSTで指定。log内の created at (UTC) をJSTに変換して判定します。")
    now_jst = datetime.now(JST)
    default_start = (now_jst.replace(hour=0, minute=0, second=0, microsecond=0))
    default_end = now_jst

    start_time = st.datetime_input("開始時間（JST）", value=default_start)
    end_time = st.datetime_input("終了時間（JST）", value=default_end)

    st.divider()
    st.subheader("ブラックリスト")
    bl_file = st.file_uploader("ブラックリストCSV（任意）", type=["csv"], key="bl_csv")
    seed_text = st.text_input("抽選シード（空ならランダム）", value="")

    run = st.button("抽選開始", type="primary")

# ブラックリスト読み込み
blacklist_df = pd.DataFrame(columns=["sender_user_id", "username"])
if bl_file is not None:
    try:
        tmp = pd.read_csv(bl_file)
        blacklist_df = normalize_blacklist(tmp)
    except Exception as e:
        st.error(f"ブラックリストCSVの読み込みに失敗: {e}")

# メイン処理
if run:
    if log_file is None:
        st.error("logファイルをアップロードしてください。")
        st.stop()

    raw = log_file.read().decode("utf-8", errors="replace")
    df_all = parse_log_to_df(raw)

    col1, col2, col3 = st.columns(3)
    col1.metric("抽出件数（requestInvite）", int(len(df_all)))
    col2.metric("ブラックリスト件数", int(len(blacklist_df)))
    col3.metric("抽選人数", int(num_winners))

    if df_all.empty:
        st.warning("requestInvite に該当する行が見つかりませんでした。")
        st.stop()

    df_time = apply_time_filter(df_all, start_time, end_time)
    df_clean = apply_blacklist(df_time, blacklist_df)
    df_participants = dedup_participants(df_clean)

    st.subheader("参加者（時間内・BL除外・重複除去後）")
    st.dataframe(df_participants, use_container_width=True, height=260)

    # seed
    seed = None
    if seed_text.strip() != "":
        try:
            seed = int(seed_text.strip())
        except ValueError:
            # 文字列でもOKにする（安定再現用）
            seed = abs(hash(seed_text.strip())) % (2**32)

    winners_df = draw_winners(df_participants, int(num_winners), seed)

    st.subheader("当選者")
    if winners_df.empty:
        st.warning("当選者が作れませんでした（参加者ゼロ or 抽選人数0）。")
    else:
        st.dataframe(winners_df[["username", "sender_user_id", "created_at_jst"]], use_container_width=True, height=240)

    # エクスポート
    st.divider()
    st.subheader("エクスポート / インポート")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.download_button(
            "当選者リストCSVをダウンロード",
            data=df_to_csv_bytes(winners_df[["username", "sender_user_id", "created_at_jst"]]),
            file_name="winners.csv",
            mime="text/csv",
        )

    with c2:
        st.download_button(
            "参加者リストCSVをダウンロード",
            data=df_to_csv_bytes(df_participants[["username", "sender_user_id", "created_at_jst"]]),
            file_name="participants.csv",
            mime="text/csv",
        )

    with c3:
        st.download_button(
            "ブラックリストCSVをダウンロード",
            data=df_to_csv_bytes(blacklist_df),
            file_name="blacklist.csv",
            mime="text/csv",
        )

else:
    st.info("左のサイドバーで log と条件を入力して「抽選開始」を押してください。")
    st.caption("パース対象は type:requestInvite を含む通知行です。created at: ... UTC をJSTへ変換します。")
