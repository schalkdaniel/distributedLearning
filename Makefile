.PHONY: all
all:
	docker build --no-cache --pull --rm -t train_iris:station.0 .

