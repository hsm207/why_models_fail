run:
	TF_CPP_MIN_LOG_LEVEL=3 python main.py --seed 1 --shuffle_data False

run-random:
	TF_CPP_MIN_LOG_LEVEL=3 python main.py

run-verbose:
	python main.py --seed 1 --shuffle_data False