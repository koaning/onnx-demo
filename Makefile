build:
	docker build -t onnxdemo .

run: build
	docker run --rm -it -p 8080:8080 onnxdemo
