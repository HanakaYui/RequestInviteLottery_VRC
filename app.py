# app.py
# Streamlit: VRChat log をドラッグ&ドロップ → type:requestInvite をパース
# 指定時間で絞り込み → 抽選人数を指定して抽選
# 参加者リスト / 当選者リスト / ブラックリスト（username, sender_user_id, reason）のインポート・エクスポート対応
#
# 実行: streamlit run app.py

from __future__ import annotations

import io
import re
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pandas as pd
import streamlit as st


# =========================
# Regex / Parsing
# =========================
TS_RE = re.compile(r"^(?P<ts>\d{4}\.\d{2}\.\d{2} \d{2}:\d{2}:\d{2})")

# 想定ログ1行:
NOTIF_RE = re.compile(
    r"Received Notification:\s*<Notification\s+"
    r"from username:(?P<username>.*?),\s*"
    r"sender user id:\s*(?P<sender>usr_[0-9a-fA-F-]+)\s+"
    r".*?\btype:\s*(?P<type>[A-Za-z0-9_]+)\b"
)


def _parse_ts(ts_str: str) -> Optional[datetime]:
    try:
        return datetime.strptime(ts_str, "%Y.%m.%d %H:%M:%S")
    except Exception:
        return None


def parse_request_invite_lines(lines: list[str]) -> pd.DataFrame:
    """
    - "Received Notification:" 行のみ対象
    - type が requestInvite で完全一致のものだけ採用
    """
    rows = []
    for line in lines:
        if "Received Notification:" not in line:
            continue

        m = NOTIF_RE.search(line)
        if not m:
            continue

        typ = (m.group("type") or "").strip()
        if typ != "requestInvite":
            continue

        ts_m = TS_RE.match(line)
        ts_str = ts_m.group("ts") if ts_m else None
        ts = _parse_ts(ts_str) if ts_str else None

        username = (m.group("username") or "").strip()
        if username == "":
            username = None

        rows.append(
            {
                "timestamp_str": ts_str,
                "timestamp": ts,
                "username": username,
                "sender_user_id": m.group("sender"),
                "raw_line": line,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.drop_duplicates(subset=["sender_user_id", "username", "timestamp_str"], keep="first")

    df = df.sort_values(
        by=["timestamp", "sender_user_id"],
        ascending=[True, True],
        na_position="last",
        kind="stable",
    ).reset_index(drop=True)

    return df


# =========================
# Blacklist (username, sender_user_id, reason)
# internal: dict[sender_user_id] = {"username": str, "reason": str}
# =========================
def normalize_text(s: str) -> str:
    return (s or "").strip()


def normalize_user_id(s: str) -> str:
    return normalize_text(s)


def blacklist_to_df(blacklist: dict[str, dict[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "username": meta.get("username", ""),
                "sender_user_id": uid,
                "reason": meta.get("reason", ""),
            }
            for uid, meta in sorted(blacklist.items())
        ]
    )


def load_blacklist_from_csv_bytes(content: bytes) -> dict[str, dict[str, str]]:
    """
    CSV想定:
      username,sender_user_id,reason
    - username/reason は任意（空OK）
    - sender_user_id は必須
    """
    df = pd.read_csv(io.BytesIO(content))
    if df.empty:
        return {}

    cols = {c.lower(): c for c in df.columns}
    if "sender_user_id" not in cols:
        raise ValueError("CSVに sender_user_id 列がありません（必須）")

    uid_col = cols["sender_user_id"]
    name_col = cols.get("username")
    reason_col = cols.get("reason")

    result: dict[str, dict[str, str]] = {}
    for _, row in df.iterrows():
        uid = normalize_user_id(str(row[uid_col]))
        if not uid:
            continue

        username = normalize_text(str(row[name_col])) if name_col else ""
        reason = normalize_text(str(row[reason_col])) if reason_col else ""

        result[uid] = {"username": username, "reason": reason}

    return result


def merge_blacklist_auto_fill(
    blacklist: dict[str, dict[str, str]],
    participants: pd.DataFrame,
) -> dict[str, dict[str, str]]:
    """
    ブラックリストの username が空のものを、participants から補完（見つかれば）
    """
    if participants is None or participants.empty:
        return blacklist

    # map sender_user_id -> username（participants優先で最初の非空）
    uid_to_name: dict[str, str] = {}
    for _, r in participants.iterrows():
        uid = str(r.get("sender_user_id", "")).strip()
        name = r.get("username", None)
        if uid and uid not in uid_to_name:
            uid_to_name[uid] = ("" if name is None else str(name).strip())

    for uid, meta in blacklist.items():
        if normalize_text(meta.get("username", "")) == "":
            cand = uid_to_name.get(uid, "")
            if cand:
                meta["username"] = cand

    return blacklist


# =========================
# Lottery
# =========================
@dataclass
class LotteryResult:
    winners: pd.DataFrame
    pool: pd.DataFrame
    excluded_blacklist: pd.DataFrame


def run_lottery(
    participants: pd.DataFrame,
    blacklist: dict[str, dict[str, str]],
    n_winners: int,
    seed: Optional[int] = None,
) -> LotteryResult:
    """
    participants: username, sender_user_id を含むDF
    blacklist: dict[sender_user_id] = meta
    """
    if participants.empty:
        return LotteryResult(
            winners=pd.DataFrame(columns=participants.columns),
            pool=participants,
            excluded_blacklist=pd.DataFrame(columns=participants.columns),
        )

    bl_keys = set(blacklist.keys())

    mask_bl = participants["sender_user_id"].astype(str).isin(bl_keys)
    excluded = participants[mask_bl].copy()
    pool = participants[~mask_bl].copy()

    if pool.empty or n_winners <= 0:
        return LotteryResult(
            winners=pd.DataFrame(columns=participants.columns),
            pool=pool,
            excluded_blacklist=excluded,
        )

    pool_unique = pool.drop_duplicates(subset=["sender_user_id"], keep="first").reset_index(drop=True)

    k = min(n_winners, len(pool_unique))
    rng = random.Random(seed)
    idxs = rng.sample(range(len(pool_unique)), k=k)
    winners = pool_unique.iloc[idxs].copy().reset_index(drop=True)

    return LotteryResult(winners=winners, pool=pool_unique, excluded_blacklist=excluded)


# =========================
# UI helpers
# =========================
def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def suggest_default_time_range(df: pd.DataFrame) -> tuple[Optional[datetime], Optional[datetime]]:
    if df.empty or "timestamp" not in df.columns:
        return None, None
    ts = df["timestamp"].dropna()
    if ts.empty:
        return None, None
    return ts.min().to_pydatetime(), ts.max().to_pydatetime()


# =========================
# Streamlit App
# =========================
def main() -> None:
    st.set_page_config(page_title="RequestInviteLottery", layout="wide")
    st.title("RequestInviteLottery")
    st.caption(
        "ログをドラッグ&ドロップ → requestInvite だけ抽出 → 指定時間で絞って抽選。"
        "ブラックリストは username / sender_user_id / reason のCSV入出力対応。"
    )

    # ---- session state ----
    if "blacklist" not in st.session_state:
        st.session_state.blacklist = {}  # dict[sender_user_id] = {"username":..., "reason":...}
    if "winners_df" not in st.session_state:
        st.session_state.winners_df = pd.DataFrame()
    if "participants_df" not in st.session_state:
        st.session_state.participants_df = pd.DataFrame()

    # =========================
    # Sidebar: Settings
    # =========================
    with st.sidebar:
        st.header("抽選設定")

        n_winners = st.number_input("抽選人数", min_value=0, max_value=10000, value=1, step=1)

        use_seed = st.checkbox("シード固定（再現性が必要な場合ON）", value=False)
        seed = None
        if use_seed:
            seed = st.number_input("seed", min_value=0, max_value=2_147_483_647, value=42, step=1)

        st.divider()
        st.subheader("ブラックリスト（username / user_id / reason）")

        # 手動追加
        add_name = st.text_input("ユーザー名（任意）", value="", key="bl_add_name")
        add_uid = st.text_input("ユーザーID（必須）（usr_...）", value="", key="bl_add_uid")
        add_reason = st.text_input("理由（任意）", value="", key="bl_add_reason")

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("追加", use_container_width=True):
                uid = normalize_user_id(add_uid)
                if uid:
                    st.session_state.blacklist[uid] = {
                        "username": normalize_text(add_name),
                        "reason": normalize_text(add_reason),
                    }
        with col_b:
            if st.button("全クリア", use_container_width=True):
                st.session_state.blacklist = {}

        # インポート
        bl_up = st.file_uploader("ブラックリストCSVをインポート", type=["csv"], key="bl_uploader")
        if bl_up is not None:
            try:
                imported = load_blacklist_from_csv_bytes(bl_up.getvalue())
                st.session_state.blacklist.update(imported)
                st.success(f"インポート: {len(imported)} 件")
            except Exception as e:
                st.error(f"ブラックリストCSVの読み取りに失敗: {e}")

        # ブラックリスト表示＆エクスポート
        bl_df = blacklist_to_df(st.session_state.blacklist)
        st.write(f"現在のブラックリスト: {len(st.session_state.blacklist)} 件")
        st.dataframe(bl_df, use_container_width=True, height=220)

        st.download_button(
            "ブラックリストCSVをエクスポート",
            data=df_to_csv_bytes(bl_df),
            file_name="blacklist.csv",
            mime="text/csv",
            use_container_width=True,
        )

        with st.expander("ブラックリストCSVフォーマット例"):
            st.code(
                "username,sender_user_id,reason\n"
                "Alice,usr_aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee,迷惑行為\n"
                "Bob,usr_ffffffff-1111-2222-3333-444444444444,視界ジャック\n"
            )

    # =========================
    # Main: Upload log & parse
    # =========================
    st.subheader("1) ログ投入（ドラッグ&ドロップ）")
    up = st.file_uploader("VRChat log (.txt / .log)", type=["txt", "log"], key="log_uploader")

    if up is None:
        st.info("ログファイルをアップロードしてください。")
        return

    raw = up.getvalue().decode("utf-8", errors="replace")
    lines = raw.splitlines()
    parsed_df = parse_request_invite_lines(lines)

    if parsed_df.empty:
        st.warning("type:requestInvite が見つかりませんでした。")
        with st.expander("デバッグ: Received Notification を含む行（先頭200行中）"):
            sample = [l for l in lines[:200] if "Received Notification" in l]
            st.text("\n".join(sample[:50]) if sample else "該当行なし")
        return

    st.success(f"requestInvite 抽出: {len(parsed_df)} 行")

    # =========================
    # 2) Time filter
    # =========================
    st.subheader("2) 指定時間で絞り込み")

    min_ts, max_ts = suggest_default_time_range(parsed_df)

    col1, col2 = st.columns(2)
    with col1:
        start_dt = st.date_input(
            "開始日",
            value=min_ts.date() if min_ts else datetime.now().date(),
        )
        start_tm = st.time_input(
            "開始時刻",
            value=min_ts.time() if min_ts else datetime.now().time().replace(second=0, microsecond=0),
        )
    with col2:
        end_dt = st.date_input(
            "終了日",
            value=max_ts.date() if max_ts else datetime.now().date(),
        )
        end_tm = st.time_input(
            "終了時刻",
            value=max_ts.time() if max_ts else datetime.now().time().replace(second=0, microsecond=0),
        )

    start = datetime.combine(start_dt, start_tm)
    end = datetime.combine(end_dt, end_tm)

    time_mask = parsed_df["timestamp"].notna() & (parsed_df["timestamp"] >= start) & (parsed_df["timestamp"] <= end)
    filtered_df = parsed_df[time_mask].copy().reset_index(drop=True)

    st.write(f"時間内 requestInvite: **{len(filtered_df)} 行**（{start} 〜 {end}）")

    participants = (
        filtered_df[["username", "sender_user_id", "timestamp_str", "timestamp"]]
        .drop_duplicates(subset=["sender_user_id"], keep="first")
        .sort_values(["timestamp", "sender_user_id"], kind="stable")
        .reset_index(drop=True)
    )

    st.session_state.participants_df = participants

    # blacklist の username を participants から補完（空欄のみ）
    st.session_state.blacklist = merge_blacklist_auto_fill(st.session_state.blacklist, participants)

    with st.expander("抽出データ（requestInvite行）"):
        st.dataframe(filtered_df[["timestamp_str", "username", "sender_user_id"]], use_container_width=True)

    st.subheader("リクインリスト")
    st.write(f"参加者: **{len(participants)} 人**")
    st.dataframe(participants[["timestamp_str", "username", "sender_user_id"]], use_container_width=True)

    st.download_button(
        "リクインリストCSVをエクスポート",
        data=df_to_csv_bytes(participants[["timestamp_str", "username", "sender_user_id"]]),
        file_name="participants.csv",
        mime="text/csv",
    )

    # =========================
    # 3) Lottery
    # =========================
    st.subheader("3) 抽選")

    col_run, col_info = st.columns([1, 2])
    with col_run:
        if st.button("抽選開始", type="primary", use_container_width=True):
            result = run_lottery(
                participants=participants[["username", "sender_user_id"]].copy(),
                blacklist=st.session_state.blacklist,
                n_winners=int(n_winners),
                seed=int(seed) if use_seed else None,
            )
            st.session_state.winners_df = result.winners
            st.session_state._last_pool_df = result.pool
            st.session_state._last_excluded_bl_df = result.excluded_blacklist

    with col_info:
        st.write(
            f"- 抽選人数: **{int(n_winners)}**\n"
            f"- シード: **{seed if use_seed else '未使用'}**\n"
            f"- ブラックリスト: **{len(st.session_state.blacklist)} 件**"
        )

    winners_df = st.session_state.winners_df
    if winners_df is not None and not winners_df.empty:
        st.success(f"当選者: **{len(winners_df)} 人**")
        st.dataframe(winners_df, use_container_width=True)

        st.download_button(
            "当選者CSVをエクスポート",
            data=df_to_csv_bytes(winners_df),
            file_name="winners.csv",
            mime="text/csv",
        )

        with st.expander("抽選詳細"):
            pool_df = getattr(st.session_state, "_last_pool_df", pd.DataFrame())
            excl_df = getattr(st.session_state, "_last_excluded_bl_df", pd.DataFrame())

            st.write(f"抽選母集団: {len(pool_df)} 人")
            st.dataframe(pool_df, use_container_width=True)

            st.write(f"ブラックリストにより除外: {len(excl_df)} 人")
            st.dataframe(excl_df, use_container_width=True)

            # 除外者に reason を付けた表
            if not excl_df.empty:
                reason_rows = []
                for _, r in excl_df.iterrows():
                    uid = str(r["sender_user_id"])
                    meta = st.session_state.blacklist.get(uid, {})
                    reason_rows.append(
                        {
                            "username": r.get("username", ""),
                            "sender_user_id": uid,
                            "reason": meta.get("reason", ""),
                        }
                    )
                st.write("除外者（理由付き）")
                st.dataframe(pd.DataFrame(reason_rows), use_container_width=True)

    else:
        st.info("「抽選開始」を押すと当選者が表示されます。")


if __name__ == "__main__":
    main()
