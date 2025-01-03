# Light Version
python src/test_real.py --dataset AIDS --light --topk 1
python src/test_real.py --dataset Linux --light --topk 1
python src/test_real.py --dataset IMDB --light --topk 1
# Fast Version
python src/test_real.py --dataset AIDS
python src/test_real.py --dataset Linux
python src/test_real.py --dataset IMDB
# Full Version
python src/test_real.py --dataset AIDS --patience 20
python src/test_real.py --dataset Linux --patience 20
python src/test_real.py --dataset IMDB --patience 20