.PHONY: all
all:
	docker build --no-cache --pull --rm -t train_iris:base .
	docker tag train_iris:base train_iris:0-station.0
	docker tag train_iris:base personalhealthtrain/train_iris:base

