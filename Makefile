.PHONY: data tokens train eval tables figs clean

data:
	bash sitpath_eval/data/download_eth_ucy.sh
	bash sitpath_eval/data/download_sdd_mini.sh || true
	@echo "See download_nuscenes_mini.md for credentials."

tokens:
	python scripts/precompute_tokens.py --dataset eth_ucy

train:
	python scripts/train.py --cfg configs/eth_ucy/sitpath_transformer.yaml

eval:
	python scripts/evaluate.py --split test

tables:
	python sitpath_eval/tables_figs/make_tables.py

figs:
	python sitpath_eval/tables_figs/make_figs.py

clean:
	rm -rf outputs/* cache/* tables/* figs/*
