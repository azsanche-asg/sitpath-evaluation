.PHONY: data tokens train eval tables figs clean
data:
\tbash sitpath_eval/data/download_eth_ucy.sh
\tbash sitpath_eval/data/download_sdd_mini.sh || true
\t@echo "See download_nuscenes_mini.md for credentials."

tokens:
\tpython scripts/precompute_tokens.py --dataset eth_ucy

train:
\tpython scripts/train.py --cfg configs/eth_ucy/sitpath_transformer.yaml

eval:
\tpython scripts/evaluate.py --split test

tables:
\tpython sitpath_eval/tables_figs/make_tables.py

figs:
\tpython sitpath_eval/tables_figs/make_figs.py

clean:
\trm -rf outputs/* cache/* tables/* figs/*
