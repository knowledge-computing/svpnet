# SVPNet
Modeling Spatially Varying Physical Dynamics for Spatiotemporal Predictive Learning [paper/SigSpatial2023_spatiotemporal_prediction.pdf](https://sigspatial2023.sigspatial.org/giscup/index.html)

## Data

## Running the code

- To train SVPNet on Moving MNIST++, run
`python train.py --data_source moving_mnist_3_tr2_ro1 --seq_len 4 --horizon 8 --model_name svpnet --batch_size 64 --gpu_id 0 --lr 0.001 --num_layers 3 --result_root ./results --num_epochs 150 --tau 3

- To train SVPNet on Sea Surface Temperature, run
`python train.py --data_source sst --seq_len 4 --horizon 6 --model_name svpnet --batch_size 64 --gpu_id 0 --lr 0.001 --num_layers 3 --result_root ./results --num_epochs 150 --tau 3
	
- To train SVPNet on TaxiBJ, run
`python train.py --data_source taxibj --seq_len 4 --horizon 4 --model_name svpnet --batch_size 64 --gpu_id 0 --lr 0.0005 --num_layers 3 --result_root ./results --num_epochs 150 --tau 3
	
- To train SVPNet on Turbulent, run
`python train.py --data_source turbulent --seq_len 10 --horizon 10 --model_name svpnet --batch_size 64 --gpu_id 0 --lr 0.001 --num_layers 3 --result_root ./results --num_epochs 150 --tau 3
	
- To test SVPNet, run
`python test.py --data_source [DATA_SOURCE] --model [MODEL] --result_path [RESULT_PATH] --gpu_id 0
	
	
	- DATA_SOURCE: Choose from [moving_mnist_3_tr2_ro1 | sst | taxibj | turbulent]
	- MODEL: The model folder generated during training
	- RESULT_PATH: The path for the model