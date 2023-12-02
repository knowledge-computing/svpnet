# SVPNet - Modeling Spatially Varying Physical Dynamics for Spatiotemporal Predictive Learning 

[Paper](paper/SigSpatial2023_spatiotemporal_prediction.pdf)



## Data

[Moving MNIST+](https://drive.google.com/drive/folders/1c9ylmDFjCKEUQG-XKkAhUr0N8IqeFjbZ?usp=drive_link)

[TaxiBJ](https://drive.google.com/drive/folders/11J8Z3fZP9M5xgZDa7AMaYJqinQY9x3Q0?usp=drive_link)

[Sea Surface Temperature](https://drive.google.com/drive/folders/18NeYaRwkYN_uNURMbNGCI53-7TqvUvpU?usp=drive_link)

[Turbulent](https://drive.google.com/drive/folders/1SKpIql11TjztGOuNPTZ2mHmKBp9R-AaE?usp=drive_link)


## Running the code

- To train SVPNet on Moving MNIST+, run
	```python train.py --data_source moving_mnist_3_tr2_ro1 --seq_len 4 --horizon 8 --model_name svpnet --batch_size 64 --gpu_id 0 --lr 0.001 --num_layers 3 --result_root ./results --num_epochs 150 --tau 3```

- To train SVPNet on TaxiBJ, run
	```python train.py --data_source taxibj --seq_len 4 --horizon 4 --model_name svpnet --batch_size 64 --gpu_id 0 --lr 0.0005 --num_layers 3 --result_root ./results --num_epochs 150 --tau 3```

- To train SVPNet on Sea Surface Temperature, run
	```python train.py --data_source sst --seq_len 4 --horizon 6 --model_name svpnet --batch_size 64 --gpu_id 0 --lr 0.001 --num_layers 3 --result_root ./results --num_epochs 150 --tau 3```
		
- To train SVPNet on Turbulent, run
	```python train.py --data_source turbulent --seq_len 10 --horizon 10 --model_name svpnet --batch_size 64 --gpu_id 0 --lr 0.001 --num_layers 3 --result_root ./results --num_epochs 150 --tau 3```
	
- To test SVPNet, run
	```python test.py --data_source [DATA_SOURCE] --model [MODEL] --result_path [RESULT_PATH] --gpu_id 0```
	
	- DATA_SOURCE: Choose from [moving_mnist_3_tr2_ro1 | sst | taxibj | turbulent]
	- MODEL: The model folder generated during training
	- RESULT_PATH: The path for the model

