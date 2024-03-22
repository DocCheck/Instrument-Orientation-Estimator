dockerContainer = rona_ori_est
dockerParams = -v $$PWD/:/project -it
dockerGPUParams = -v $$PWD/:/project -it --gpus all


ifeq ($(env), ci)
	dockerParams =
endif


default:
	@echo "hello world"

build:
	docker build -t $(dockerContainer) .

bash:
	docker run $(dockerParams) $(dockerContainer) bash


train-image:
	docker run $(dockerGPUParams) $(dockerContainer) python3 -m main --input-data "data/dataset/Rona_dataset_rot_est_all/" --model-path "data/models/model_default/Rona_ori_est_model.pt" --n-class 357 --epochs 1000 train-model

predict-image:
	docker run $(dockerGPUParams) $(dockerContainer) python3 -m main --input-data "data/dataset/Rona_dataset_ori_est_test/single/" --model-path "data/models/model_default/Rona_ori_est_model.pt" --diff-t 3 predict-model