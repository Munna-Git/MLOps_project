build:
	docker build -t mlops_project:latest .

run:
	docker run -it --rm -p 5000:5000 ^
        -v D:/mlops_project/data:/app/data ^
        -v D:/mlops_project/models:/app/models ^
        mlops_project

logs:
	docker ps -a
