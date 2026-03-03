#!/usr/bin/env python3
"""Preprocess Goodreads data for task 1."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_user_id_mapping(user_ids: pd.Series, start_id: int = 1_000_000) -> dict[str, int]:
    unique_sorted = sorted(user_ids.astype(str).unique())
    return {uid: start_id + idx for idx, uid in enumerate(unique_sorted)}


def preprocess(
    books_path: Path,
    interactions_path: Path,
    items_out_path: Path,
    events_out_path: Path,
) -> tuple[int, int, float]:
    items = pd.read_parquet(books_path)
    interactions = pd.read_parquet(interactions_path)

    total_events = len(interactions)
    events = interactions.copy()

    cutoff = pd.Timestamp("2017-11-01")
    events["started_at"] = pd.to_datetime(events["started_at"], errors="coerce")
    events["read_at"] = pd.to_datetime(events["read_at"], errors="coerce")

    events = events[
        (events["started_at"] < cutoff) & (events["read_at"] < cutoff)
    ].copy()
    events = events[events["rating"].notna() & (events["rating"] > 0)].copy()
    events = events[events["is_read"] == True].copy()

    user_reads = events.groupby("user_id").size()
    valid_users = user_reads[user_reads >= 2].index
    events = events[events["user_id"].isin(valid_users)].copy()

    items = items.rename(columns={"book_id": "item_id"})
    events = events.rename(columns={"book_id": "item_id"})

    user_mapping = build_user_id_mapping(events["user_id"], start_id=1_000_000)
    events["user_id"] = events["user_id"].map(user_mapping).astype("int64")

    items.to_parquet(items_out_path, index=False)
    events.to_parquet(events_out_path, index=False)

    remaining_events = len(events)
    share = remaining_events / total_events if total_events else 0.0
    return total_events, remaining_events, share


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 1 preprocessing script.")
    parser.add_argument(
        "--books-path",
        type=Path,
        default=Path("goodsread/books.parquet"),
        help="Path to books.parquet",
    )
    parser.add_argument(
        "--interactions-path",
        type=Path,
        default=Path("goodsread/interactions.parquet"),
        help="Path to interactions.parquet",
    )
    parser.add_argument(
        "--items-out-path",
        type=Path,
        default=Path("items.parquet"),
        help="Output path for items.parquet",
    )
    parser.add_argument(
        "--events-out-path",
        type=Path,
        default=Path("events.parquet"),
        help="Output path for events.parquet",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total_events, remaining_events, share = preprocess(
        books_path=args.books_path,
        interactions_path=args.interactions_path,
        items_out_path=args.items_out_path,
        events_out_path=args.events_out_path,
    )
    print(f"Total events: {total_events}")
    print(f"Remaining events: {remaining_events}")
    print(f"Share remaining: {share:.6f}")


if __name__ == "__main__":
    main()
