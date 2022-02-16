install:
	python -m pip install -r requirements.txt

build:
	docker build -t onnxdemo .

run: build
	docker run --rm -it -p 8080:8080 onnxdemo

benchmark-web:
	python -m locust -f locust-benchmark.py --headless --host http://0.0.0.0:8080 -u 100 -r 1

benchmark:
	python benchmark.py
